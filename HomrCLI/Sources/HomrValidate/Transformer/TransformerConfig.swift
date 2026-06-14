import Foundation

// Port of `homr/transformer/configs.py` (`Config` + `DecoderArgs`).
//
// Only the fields that matter for ONNX *inference* are kept. Training-only
// knobs (scheduled sampling, dropout schedules, backbone layer counts, encoder
// structure, etc.) are dropped because they have no effect once the network is
// exported to ONNX — the graph already bakes those choices in.
//
// The public surface here is depended upon EXACTLY by `Encoder`, `ScoreDecoder`
// and `Staff2Score`, so do not rename/retype these fields without updating them.

/// Subset of `DecoderArgs` from the Python config.
///
/// These flags only influenced how the PyTorch decoder was *built* before being
/// traced to ONNX; at inference time the ONNX graph is fixed, so they are purely
/// informational here. Kept as an analogue for parity/documentation.
final class DecoderArgs {
    let attnOnAttn = true
    let crossAttend = true
    let ffGlu = true
    let relPosBias = false
    let useScalenorm = false
    // Dropout is a no-op at inference; recorded for fidelity with the Python config.
    let attnDropout = 0.1
    let ffDropout = 0.1
    let layerDropout = 0.1
}

/// Inference configuration for the homr transformer (TrOMR).
///
/// Mirrors the Python `Config`. Constants match the values the ONNX checkpoint
/// `pytorch_model_396-…` was exported with — changing them desynchronises the
/// Swift driver from the model graph and produces garbage.
final class TransformerConfig {
    /// The six token↔index sub-vocabularies (rhythm/lift/articulation/pitch/slur/position)
    /// plus their inverses, used to detokenize decoder argmax indices.
    let vocab: Vocabulary

    /// Image / patch geometry (the encoder expects a single-channel canvas).
    let channels = 1
    let patchSize = 16
    /// Canvas the staff image is centered into before normalization: `[1,1,256,1280]`.
    let maxHeight = 256
    let maxWidth = 1280

    /// Maximum number of autoregressive decoder steps.
    let maxSeqLen = 608

    /// Special token ids (indices into the rhythm vocabulary).
    let padToken = 0
    let bosToken = 1
    let eosToken = 2
    /// "No note" / absent-field token id, shared by pitch/lift/articulation/slur.
    let nonoteToken = 0

    /// Decoder dimensions — `decoderDim / decoderHeads == 64` is the KV-cache head dim.
    let decoderDim = 512
    let decoderDepth = 8
    let decoderHeads = 8

    /// Decoder build flags (informational at inference, see `DecoderArgs`).
    let decoderArgs = DecoderArgs()

    /// Per-head dimension of the KV cache (`decoderDim / decoderHeads`). The cache
    /// tensors are shaped `[1, decoderHeads, cacheLen, headDim]`.
    var headDim: Int { decoderDim / decoderHeads }

    /// Number of KV-cache tensors threaded through the decoder: `decoderDepth * 4`
    /// (self-attn K/V + cross-attn K/V per layer = 32 for depth 8).
    var cacheTensorCount: Int { decoderDepth * 4 }

    /// On iOS there is no CUDA, so GPU inference is unavailable. The decoder always
    /// runs fp32 on the CPU execution provider; the encoder may optionally use CoreML.
    var useGpuInference = false

    /// CoreML (MLProgram) acceleration for the *encoder* (the conv-heavy forward
    /// pass run once per staff). Cost: a one-time MLProgram compile (~tens of
    /// seconds) on first use; CoreML caches the compiled model on disk afterward.
    var useCoremlEncoder = true

    init() {
        self.vocab = Vocabulary()
    }
}
