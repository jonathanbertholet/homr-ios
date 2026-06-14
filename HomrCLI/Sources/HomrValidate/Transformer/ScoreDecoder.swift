import Foundation
import OnnxRuntimeBindings

/// Port of `homr/transformer/decoder_inference.py` (`ScoreDecoder`, `init_cache`,
/// `detokenize`, `get_decoder`).
///
/// The decoder is an autoregressive **greedy** loop (argmax, no sampling) over
/// up to `maxSeqLen` steps. It emits six parallel token streams per step —
/// rhythm, pitch, lift, articulation, slur, position — that are detokenized into
/// `EncodedSymbol`s.
///
/// ## KV-cache flow (the part that makes this fast)
/// The decoder is exported with an external key/value cache. There are
/// `decoderDepth * 4 = 32` cache tensors (self-attn K/V + cross-attn K/V per
/// layer), each shaped `[1, decoderHeads(8), cacheLen, headDim(64)]`. They start
/// at `cacheLen = 0` (empty) and grow by one along the `cacheLen` axis every
/// step. Each step we feed the previous step's `cache_out{i}` back in as
/// `cache_in{i}`, so attention only recomputes the newest position.
///
/// ## Per-step decoder I/O contract (assumed — verify against the ONNX model)
/// Inputs:
///   - `rhythms`, `pitchs`, `lifts`, `articulations`, `slurs`: `[1,1]` int64,
///      the LAST sampled token of each stream (start tokens on step 0). NOTE the
///      Python typo `pitchs` (not `pitchies`) is preserved as the tensor name.
///   - `context`: `[1, seq, decoderDim]` fp32 on step 0, then the reduced
///      `context[:, :1]` (`[1, 1, decoderDim]`) on every later step. (x_transformers
///      slices `[:, :0]` internally which broke the ONNX Reshape, hence `[:, :1]`.)
///   - `cache_len`: `[1]` int64 = `[step]` (current cache length).
///   - `cache_in0` … `cache_in{4*decoderDepth-1}`: fp32 `[1,8,cacheLen,64]`.
/// Outputs (ORDER MATTERS — matches Python `outputs[0..6]`):
///   - `out_rhythms`, `out_pitchs`, `out_lifts`, `out_positions`,
///     `out_articulations`, `out_slurs`: each `[1, T, vocabSize]` fp32 logits.
///     (Note positions/articulations are swapped relative to the input order.)
///   - `attention`: attention map (used only for optional debug coordinates).
///   - `cache_out0` … `cache_out{4*decoderDepth-1}`: the next step's cache inputs.
///
/// Differences from Python: Python uses `io_binding`; `OnnxRuntimeBindings` has
/// no io-binding convenience, so each step we assemble an `[String: ORTValue]`
/// of all inputs, call `session.run(...)`, then carry the 32 cache outputs
/// (re-wrapped as fresh `ORTValue`s) into the next step. Everything stays fp32.
final class ScoreDecoder {
    private let session: ORTSession
    private let config: TransformerConfig

    /// Decoder output tensor names, in the exact order Python reads `outputs[0..6]`.
    private let logitOutputNames = [
        "out_rhythms",
        "out_pitchs",
        "out_lifts",
        "out_positions",
        "out_articulations",
        "out_slurs",
        "attention",
    ]

    /// `cache_in{i}` / `cache_out{i}` names, generated once from `cacheTensorCount`.
    private let cacheInputNames: [String]
    private let cacheOutputNames: [String]

    init(session: ORTSession, config: TransformerConfig) {
        self.session = session
        self.config = config
        self.cacheInputNames = (0..<config.cacheTensorCount).map { "cache_in\($0)" }
        self.cacheOutputNames = (0..<config.cacheTensorCount).map { "cache_out\($0)" }
    }

    /// Greedy autoregressive decode.
    ///
    /// Mirrors `ScoreDecoder.generate`. Because only the LAST token of each
    /// stream is fed back, we just track the last token per branch (no need to
    /// keep the full running sequences).
    ///
    /// - Parameters:
    ///   - startToken: rhythm BOS token id (Python passes `[[1]]`).
    ///   - nonoteToken: "no note" id for pitch/lift/articulation/slur (Python `[[0]]`).
    ///   - context: flat fp32 encoder context.
    ///   - contextShape: `[1, seq, decoderDim]`.
    /// - Returns: the decoded `EncodedSymbol`s (BOS/EOS/PAD never emitted).
    func generate(
        startToken: Int = 1,
        nonoteToken: Int = 0,
        context: [Float],
        contextShape: [Int]
    ) -> [EncodedSymbol] {
        var symbols: [EncodedSymbol] = []

        // Last sampled token per input branch; seeded with the start tokens.
        var lastRhythm = Int64(startToken)
        var lastPitch = Int64(nonoteToken)
        var lastLift = Int64(nonoteToken)
        var lastArticulation = Int64(nonoteToken)
        var lastSlur = Int64(nonoteToken)

        do {
            // Pre-wrap the full context (step 0) and the reduced context[:, :1]
            // (later steps). decoderDim is the trailing axis of the context.
            let decoderDim = contextShape.count >= 3 ? contextShape[2] : config.decoderDim
            let fullContext = try makeFloatTensor(context, shape: contextShape)
            let reducedShape = [contextShape[0], 1, decoderDim]
            let reduced = Array(context.prefix(reducedShape.reduce(1, *)))
            let reducedContext = try makeFloatTensor(reduced, shape: reducedShape)

            // Cache starts empty: 32 zero tensors of shape [1, heads, 0, headDim].
            var cache = try initCache()

            for step in 0..<config.maxSeqLen {
                // Each step is wrapped in an autorelease pool. `session.run` and
                // `tensorData()` hand back AUTORELEASED ObjC objects (the outputs
                // dictionary, ~39 ORTValues, their NSData backing). The decode loop
                // has no enclosing pool, so without this they accumulate for the
                // entire loop and balloon RSS to multiple GB — enough to get the
                // process OOM-killed on dense scores. Draining per step keeps memory
                // flat. The `cache` ORTValues we carry forward are held by a strong
                // Swift reference, so they survive the drain.
                //
                // The closure returns `false` to stop the loop (the body's former
                // `break`s become `return false`, since you can't `break` out of a
                // closure).
                let keepGoing: Bool = try autoreleasepool { () -> Bool in
                    // Assemble all inputs for this step.
                    // IMPORTANT: the ONNX export (training/onnx/convert.py) traces the
                    // decoder with positional args (..., articulations, slurs, ...) but
                    // declares input_names as (..., "slurs", "articulations", ...). The
                    // names are therefore swapped relative to the embeddings they feed:
                    // the input NAMED "articulations" actually drives slur_emb (size 5)
                    // and "slurs" drives articulation_emb (size 54). We compensate by
                    // binding the slur feedback to "articulations" and the articulation
                    // feedback to "slurs". Binding straight (as the Python reference
                    // does) overflows the size-5 slur embedding the moment a real
                    // articulation (index > 4) is predicted — which crashes on scores
                    // with ornaments/articulations (e.g. trills) and only stays dormant
                    // on simple test images. The OUTPUTS keep their correct names/sizes.
                    var inputs: [String: ORTValue] = [
                        "rhythms": try makeTokenTensor(lastRhythm),
                        "pitchs": try makeTokenTensor(lastPitch),
                        "lifts": try makeTokenTensor(lastLift),
                        "articulations": try makeTokenTensor(lastSlur),
                        "slurs": try makeTokenTensor(lastArticulation),
                        "context": step == 0 ? fullContext : reducedContext,
                        "cache_len": try makeInt64Tensor([Int64(step)], shape: [1]),
                    ]
                    for (name, value) in zip(cacheInputNames, cache) {
                        inputs[name] = value
                    }

                    // Run the decoder. We request the 7 logit/attention outputs plus
                    // the 32 cache outputs so we can thread them into the next step.
                    let outputNames = logitOutputNames + cacheOutputNames
                    // Pass nil runOptions — see app copy for why arena-shrink is off on iOS.
                    let outputs = try session.run(
                        withInputs: inputs,
                        outputNames: Set(outputNames),
                        runOptions: nil
                    )

                    // Robustness: if the model returns nothing usable, stop cleanly.
                    guard let rhythmOut = outputs["out_rhythms"] else { return false }

                    // Greedy: argmax over the last timestep's logits for each branch.
                    let rhythmSample = try argmaxLastTimestep(rhythmOut)
                    guard rhythmSample >= 0 else { return false }

                    // EOS terminates BEFORE emitting a symbol (matches Python).
                    if rhythmSample == config.eosToken { return false }

                    let pitchSample = try argmaxLastTimestep(outputs["out_pitchs"])
                    let liftSample = try argmaxLastTimestep(outputs["out_lifts"])
                    let positionSample = try argmaxLastTimestep(outputs["out_positions"])
                    let articulationSample = try argmaxLastTimestep(outputs["out_articulations"])
                    let slurSample = try argmaxLastTimestep(outputs["out_slurs"])

                    // Detokenize via inverse vocab maps.
                    let rhythmTok = detokenize(rhythmSample, config.vocab.rhythmInverse)
                    let pitchTok = detokenize(pitchSample, config.vocab.pitchInverse)
                    let liftTok = detokenize(liftSample, config.vocab.liftInverse)
                    let articulationTok = detokenize(articulationSample, config.vocab.articulationInverse)
                    let slurTok = detokenize(slurSample, config.vocab.slurInverse)
                    let positionTok = detokenize(positionSample, config.vocab.positionInverse)

                    // Any unmappable index (shouldn't happen) → stop rather than crash.
                    guard let rhythm = rhythmTok, let pitch = pitchTok, let lift = liftTok,
                          let articulation = articulationTok, let slur = slurTok,
                          let position = positionTok else {
                        return false
                    }

                    // coordinates: Python stores the raw `attention` array here, but
                    // downstream only uses it as optional debug/ordering hints. We set
                    // it to nil — downstream tolerates nil (documented deviation).
                    symbols.append(EncodedSymbol(
                        rhythm,
                        pitch: pitch,
                        lift: lift,
                        articulation: articulation,
                        slur: slur,
                        position: position,
                        coordinates: nil
                    ))

                    // Feed only the newly sampled tokens next step.
                    lastRhythm = Int64(rhythmSample)
                    lastPitch = Int64(pitchSample)
                    lastLift = Int64(liftSample)
                    lastArticulation = Int64(articulationSample)
                    lastSlur = Int64(slurSample)

                    // Thread cache_out{i} → cache_in{i} for the next iteration,
                    // re-wrapping each as a fresh fp32 ORTValue.
                    cache = try cacheOutputNames.map { name -> ORTValue in
                        guard let out = outputs[name] else {
                            throw DecoderError.missingCacheOutput(name)
                        }
                        return try rewrapFloatTensor(out)
                    }
                    return true
                }
                if !keepGoing { break }
            }
        } catch {
            // Mirror Python's "best effort" behaviour: surface a diagnostic and
            // return whatever was decoded before the failure.
            eprint("ScoreDecoder.generate failed:", error.localizedDescription)
        }

        return symbols
    }

    // MARK: - Cache

    /// Port of `init_cache`: 32 zero tensors of shape `[1, heads, 0, headDim]`
    /// (fp32 on iOS). `cacheLen` starts at 0.
    private func initCache() throws -> [ORTValue] {
        let shape = [1, config.decoderHeads, 0, config.headDim]
        // Zero-length tensor: no elements, but ONNX still needs the typed shape.
        return try (0..<config.cacheTensorCount).map { _ in
            try makeFloatTensor([], shape: shape)
        }
    }

    // MARK: - Tensor helpers

    /// Wraps a single int64 token as a `[1,1]` tensor (rhythms/pitchs/…).
    private func makeTokenTensor(_ token: Int64) throws -> ORTValue {
        try makeInt64Tensor([token], shape: [1, 1])
    }

    /// Builds an int64 ONNX tensor from a Swift array (Data(buffer:) pattern).
    private func makeInt64Tensor(_ values: [Int64], shape: [Int]) throws -> ORTValue {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return try ORTValue(
            tensorData: NSMutableData(data: data),
            elementType: .int64,
            shape: shape.map { NSNumber(value: $0) }
        )
    }

    /// Builds an fp32 ONNX tensor from a Swift array.
    private func makeFloatTensor(_ values: [Float], shape: [Int]) throws -> ORTValue {
        // Empty buffers (zero-length cache) still need valid backing storage.
        let data = values.isEmpty ? Data() : values.withUnsafeBufferPointer { Data(buffer: $0) }
        return try ORTValue(
            tensorData: NSMutableData(data: data),
            elementType: .float,
            shape: shape.map { NSNumber(value: $0) }
        )
    }

    /// Re-wraps an fp32 output tensor (e.g. `cache_out{i}`) as a fresh input
    /// `ORTValue`, copying its data and preserving its shape.
    private func rewrapFloatTensor(_ value: ORTValue) throws -> ORTValue {
        let shape = try value.tensorTypeAndShapeInfo().shape
        let data = try value.tensorData() as Data
        return try ORTValue(
            tensorData: NSMutableData(data: data),
            elementType: .float,
            shape: shape
        )
    }

    /// Argmax over the last timestep of a `[1, T, vocabSize]` fp32 logits tensor.
    ///
    /// Mirrors `outputs[k][:, -1, :].argmax()`: slice the final timestep
    /// `[1, vocabSize]`, then take the index of the maximum logit. Returns -1 if
    /// the tensor is missing or malformed (caller treats that as "stop").
    private func argmaxLastTimestep(_ value: ORTValue?) throws -> Int {
        guard let value = value else { return -1 }
        let shape = try value.tensorTypeAndShapeInfo().shape.map { $0.intValue }
        guard shape.count == 3 else { return -1 }
        let seq = shape[1]
        let vocab = shape[2]
        guard seq > 0, vocab > 0 else { return -1 }

        let data = try value.tensorData() as Data
        return data.withUnsafeBytes { raw -> Int in
            let floats = raw.bindMemory(to: Float.self)
            // Offset of the last timestep within the [1, T, V] buffer.
            let base = (seq - 1) * vocab
            var bestIndex = 0
            var bestScore = floats[base]
            for i in 1..<vocab {
                let score = floats[base + i]
                if score > bestScore {
                    bestScore = score
                    bestIndex = i
                }
            }
            return bestIndex
        }
    }

    /// Port of `detokenize` for a single index.
    ///
    /// Maps an index to its token via the inverse vocab and drops the control
    /// tokens `[BOS]`/`[EOS]`/`[PAD]` (bug-for-bug: the homr vocab actually spells
    /// these `BOS`/`EOS`/`PAD`, so those literals never match — but EOS is already
    /// handled by the early break, and BOS/PAD aren't sampled in practice).
    /// Returns nil if the index is out of range or maps to a dropped control token.
    private func detokenize(_ index: Int, _ inverse: [Int: String]) -> String? {
        guard let token = inverse[index] else { return nil }
        if token == "[BOS]" || token == "[EOS]" || token == "[PAD]" { return nil }
        return token
    }

    enum DecoderError: LocalizedError {
        case missingCacheOutput(String)
        var errorDescription: String? {
            switch self {
            case .missingCacheOutput(let name):
                return "Decoder produced no \(name) tensor"
            }
        }
    }
}

/// Port of `get_decoder`: builds the fp32 CPU decoder session (no CUDA on iOS).
///
/// `useGpuInference` is always false on iOS, so we always load the fp32 decoder
/// on the CPU execution provider via the shared factory.
func get_decoder(config: TransformerConfig) throws -> ScoreDecoder {
    let session = try ONNXSessionFactory.makeDecoderSession()
    return ScoreDecoder(session: session, config: config)
}
