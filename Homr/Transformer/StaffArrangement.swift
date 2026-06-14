import Foundation

// MARK: - Per-staff parsing & arrangement
//
// homr collapses an entire page into voices by indexing `MultiStaff.staffs[voice]`
// and, crucially, *flattens everything into a single voice* whenever the systems
// don't all contain the same number of staves (`ensureSameNumberOfStaffs`). On
// ensemble scores (e.g. 2 oboes + basso + continuo) one mis-grouped system is
// enough to trigger that fallback, so every instrument ends up concatenated into
// one sequential timeline — which is why playback was sequential, not parallel.
//
// This file replaces that with a two-step model:
//   1. `parseIndividualStaves` runs the transformer once per *physical* staff and
//      keeps each staff's symbols + geometry + OCR label (a `ParsedStaff`).
//   2. An editable `Arrangement` maps staves → parts; `assembleParts` builds one
//      parallel voice per part from the cached streams.
//
// Because the per-staff streams are cached, the arrangement can be re-edited
// (reassign instruments, regroup staves) and re-assembled without re-running the
// (expensive) transformer.

/// One physical staff after recognition: its symbols plus where it sits and what
/// instrument label (if any) was printed beside it. `Codable` so it can be saved
/// in the score library and reloaded for editing.
struct ParsedStaff: Codable, Sendable, Identifiable {
    /// Stable id within a single scan (assignment index).
    let id: Int
    /// Which system (top-to-bottom band of staves) this staff belongs to.
    /// In a multi-page scan this is GLOBAL (offset across pages) so systems stay
    /// in reading order.
    let systemIndex: Int
    /// Vertical order within its system (0 = topmost staff of the system).
    let orderInSystem: Int
    /// Which source page (0-based) this staff came from. Retained so the score
    /// can later be re-paginated exactly like the original document (drives the
    /// `<print new-page="yes"/>` breaks emitted in the MusicXML).
    let pageIndex: Int
    /// Staff bounding box in preprocessed-image coordinates.
    let minX: Double
    let minY: Double
    let maxX: Double
    let maxY: Double
    /// OCR'd instrument label from the left margin (first system only), if any.
    let label: String?
    /// The recognised symbol stream for this staff (no trailing `newline`).
    let symbols: [EncodedSymbol]
}

// Backward-compatible decoding: scores saved before `pageIndex` existed won't
// carry that key, so decode it leniently (default 0 → single page). Implemented
// in an extension so the memberwise initializer is preserved.
extension ParsedStaff {
    private enum CodingKeys: String, CodingKey {
        case id, systemIndex, orderInSystem, pageIndex, minX, minY, maxX, maxY, label, symbols
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(Int.self, forKey: .id)
        self.systemIndex = try c.decode(Int.self, forKey: .systemIndex)
        self.orderInSystem = try c.decode(Int.self, forKey: .orderInSystem)
        self.pageIndex = try c.decodeIfPresent(Int.self, forKey: .pageIndex) ?? 0
        self.minX = try c.decode(Double.self, forKey: .minX)
        self.minY = try c.decode(Double.self, forKey: .minY)
        self.maxX = try c.decode(Double.self, forKey: .maxX)
        self.maxY = try c.decode(Double.self, forKey: .maxY)
        self.label = try c.decodeIfPresent(String.self, forKey: .label)
        self.symbols = try c.decode([EncodedSymbol].self, forKey: .symbols)
    }

    func encode(to encoder: Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(systemIndex, forKey: .systemIndex)
        try c.encode(orderInSystem, forKey: .orderInSystem)
        try c.encode(pageIndex, forKey: .pageIndex)
        try c.encode(minX, forKey: .minX)
        try c.encode(minY, forKey: .minY)
        try c.encode(maxX, forKey: .maxX)
        try c.encode(maxY, forKey: .maxY)
        try c.encodeIfPresent(label, forKey: .label)
        try c.encode(symbols, forKey: .symbols)
    }
}

/// Maps recognised staves onto parts (instruments) for export and playback.
///
/// A "part" is one MusicXML `<part>` / one synth channel. By default part `p`
/// gathers the `p`-th staff (top-to-bottom) of every system, which is the
/// correct reading for a consistent ensemble score; the editor can override any
/// staff's part, the per-part instrument, and which parts are muted.
struct Arrangement: Codable, Sendable, Equatable {
    /// Number of parts (parallel voices).
    var partCount: Int
    /// `ParsedStaff.id` → part index. A value `< 0` excludes the staff.
    var staffToPart: [Int: Int]
    /// Part index → identified/assigned instrument.
    var partInstruments: [Int: Instrument]
    /// Parts the user has muted (skipped during playback).
    var mutedParts: Set<Int>
    /// When true, parts play one-after-another instead of simultaneously (useful
    /// to audition each voice, or when the "voices" are really stacked systems).
    var playSequentially: Bool = false

    /// Build the default arrangement from recognised staves: part = staff order
    /// within its system, part count = the most common staves-per-system, and
    /// instruments taken from the first system's OCR labels.
    static func makeDefault(_ staves: [ParsedStaff]) -> Arrangement {
        guard !staves.isEmpty else {
            return Arrangement(partCount: 1, staffToPart: [:], partInstruments: [:], mutedParts: [])
        }

        // Most common number of staves per system (ties → larger count).
        var perSystem: [Int: Int] = [:]
        for staff in staves { perSystem[staff.systemIndex, default: 0] += 1 }
        let partCount = modeOfCounts(Array(perSystem.values))

        // Assign each staff to the part matching its order (clamped into range).
        var staffToPart: [Int: Int] = [:]
        for staff in staves {
            staffToPart[staff.id] = min(staff.orderInSystem, partCount - 1)
        }

        // Instruments from the first system's labels.
        var partInstruments: [Int: Instrument] = [:]
        for staff in staves where staff.systemIndex == 0 {
            guard let label = staff.label, let instrument = InstrumentCatalog.match(label: label) else { continue }
            partInstruments[min(staff.orderInSystem, partCount - 1)] = instrument
        }

        return Arrangement(
            partCount: max(1, partCount),
            staffToPart: staffToPart,
            partInstruments: partInstruments,
            mutedParts: []
        )
    }

    /// Re-derive an arrangement by FORCING a fixed number of voices per system.
    ///
    /// This ignores the detector's (sometimes inconsistent) system grouping and
    /// instead reads the page purely top-to-bottom: every run of `voiceCount`
    /// consecutive staves is one system, the staff's position within that run is
    /// its part, and parts play in parallel while systems play in sequence. This
    /// is the robust manual override for ensemble scores where bracket detection
    /// miscounted (the cause of the "Voices: 1" collapse).
    static func forVoiceCount(_ voiceCount: Int, staves: [ParsedStaff]) -> Arrangement {
        let n = max(1, voiceCount)
        // Global reading order: top-to-bottom, then left-to-right.
        let ordered = staves.sorted {
            $0.minY != $1.minY ? $0.minY < $1.minY : $0.minX < $1.minX
        }

        var staffToPart: [Int: Int] = [:]
        var partInstruments: [Int: Instrument] = [:]
        for (index, staff) in ordered.enumerated() {
            let part = index % n
            staffToPart[staff.id] = part
            // Take instruments from the first system's labels (first n staves).
            if index < n, partInstruments[part] == nil,
               let label = staff.label, let instrument = InstrumentCatalog.match(label: label) {
                partInstruments[part] = instrument
            }
        }

        return Arrangement(
            partCount: n,
            staffToPart: staffToPart,
            partInstruments: partInstruments,
            mutedParts: []
        )
    }

    /// Most frequent value; on ties the larger value wins. 1 for empty input.
    private static func modeOfCounts(_ counts: [Int]) -> Int {
        guard !counts.isEmpty else { return 1 }
        var frequency: [Int: Int] = [:]
        for c in counts { frequency[c, default: 0] += 1 }
        return frequency.max { a, b in
            a.value != b.value ? a.value < b.value : a.key < b.key
        }!.key
    }
}

// Backward-compatible decoding: scores saved before `playSequentially` existed
// won't carry that key, so decode it leniently (default false). Implemented in an
// extension so the memberwise initializer is preserved.
extension Arrangement {
    private enum CodingKeys: String, CodingKey {
        case partCount, staffToPart, partInstruments, mutedParts, playSequentially
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.partCount = try c.decode(Int.self, forKey: .partCount)
        self.staffToPart = try c.decode([Int: Int].self, forKey: .staffToPart)
        self.partInstruments = try c.decode([Int: Instrument].self, forKey: .partInstruments)
        self.mutedParts = try c.decode(Set<Int>.self, forKey: .mutedParts)
        self.playSequentially = try c.decodeIfPresent(Bool.self, forKey: .playSequentially) ?? false
    }

    func encode(to encoder: Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(partCount, forKey: .partCount)
        try c.encode(staffToPart, forKey: .staffToPart)
        try c.encode(partInstruments, forKey: .partInstruments)
        try c.encode(mutedParts, forKey: .mutedParts)
        try c.encode(playSequentially, forKey: .playSequentially)
    }
}

/// Assemble per-part voices (and their instruments) from cached staff streams.
///
/// Each part concatenates its assigned staves in system order, separated by a
/// `newline` symbol (as the transformer pipeline expects), then de-duplicates.
/// The returned arrays are aligned: `voices[p]` is played/exported as part `p`
/// with `instruments[p]`.
func assembleParts(_ staves: [ParsedStaff], _ arrangement: Arrangement)
    -> (voices: [[EncodedSymbol]], instruments: [Instrument?]) {
    var byPart: [Int: [ParsedStaff]] = [:]
    for staff in staves {
        guard let part = arrangement.staffToPart[staff.id], part >= 0, part < arrangement.partCount else {
            continue
        }
        byPart[part, default: []].append(staff)
    }

    var voices: [[EncodedSymbol]] = []
    var instruments: [Instrument?] = []
    for part in 0..<arrangement.partCount {
        let partStaves = (byPart[part] ?? []).sorted { $0.systemIndex < $1.systemIndex }
        var symbols: [EncodedSymbol] = []
        for staff in partStaves where !staff.symbols.isEmpty {
            symbols.append(contentsOf: staff.symbols)
            symbols.append(EncodedSymbol("newline"))
        }
        voices.append(remove_duplicated_symbols(symbols))
        instruments.append(arrangement.partInstruments[part])
    }
    return (voices, instruments)
}

// MARK: - Score assembly

/// A fully assembled score: serialised MusicXML plus the playable/exportable
/// MIDI derived from it. Produced from cached staves + an arrangement, so the
/// editor can regenerate everything after a change without re-running the model.
struct AssembledScore: Sendable {
    let musicXML: String
    let midiSequence: MIDISequence?
    let midiData: Data?
}

/// Build MusicXML + MIDI for a given arrangement of already-recognised staves.
///
/// Muted parts are removed from the playable/exportable MIDI (their channel is
/// dropped) but remain present in the MusicXML so the notation/editor still shows
/// them.
func buildScore(from staves: [ParsedStaff], arrangement: Arrangement, title: String = "") -> AssembledScore {
    let (voices, instruments) = assembleParts(staves, arrangement)
    let document = generateXml(
        XmlGeneratorArguments(), staffs: voices, title: title, instruments: instruments
    )
    let xmlString = document.xmlString()

    var sequence = MusicXMLMIDIConverter.convert(document)
    if !arrangement.mutedParts.isEmpty {
        // Channel == part index in the converter, so muted parts map directly.
        sequence.notes.removeAll { arrangement.mutedParts.contains($0.channel) }
    }
    if arrangement.playSequentially {
        sequence = sequentialized(sequence)
    }
    let isEmpty = sequence.isEmpty
    return AssembledScore(
        musicXML: xmlString,
        midiSequence: isEmpty ? nil : sequence,
        midiData: isEmpty ? nil : StandardMIDIFile.data(from: sequence)
    )
}

/// Offset each part (channel) so they play one after another instead of
/// simultaneously. A one-beat gap separates consecutive parts.
private func sequentialized(_ sequence: MIDISequence) -> MIDISequence {
    var result = sequence
    let channels = Set(sequence.notes.map(\.channel)).sorted()
    var offset = 0.0
    var moved: [MIDINote] = []
    for channel in channels {
        let partNotes = sequence.notes.filter { $0.channel == channel }
        let end = partNotes.map { $0.startBeats + $0.durationBeats }.max() ?? 0
        for var note in partNotes {
            note.startBeats += offset
            moved.append(note)
        }
        offset += end + 1.0   // 1-beat breath between parts
    }
    result.notes = moved
    return result
}

/// Run the transformer once per physical staff, keeping each staff separate.
///
/// Systems are taken from the detector's `MultiStaff` grouping (its bracket /
/// grand-staff connections) in reading order; within a system staves are already
/// sorted top-to-bottom. Unlike `parseStaffs`, nothing is flattened or dropped on
/// inconsistent staff counts, so ensemble parts stay independent.
func parseIndividualStaves(
    staffs: [MultiStaff],
    image: GrayscaleImage,
    labelImage: GrayscaleImage,
    config: TransformerConfig
) -> [ParsedStaff] {
    // Systems in reading order (topmost first).
    let systems = staffs.sorted {
        ($0.staffs.first?.minY ?? .greatestFiniteMagnitude)
            < ($1.staffs.first?.minY ?? .greatestFiniteMagnitude)
    }
    let regions = StaffRegions(staffs)

    // Flatten every staff into a single ordered task list. The flattened index
    // doubles as the `ParsedStaff.id` and the transformer's `index`, so the
    // result is byte-for-byte identical to the old nested-loop order.
    struct StaffTask { let systemIndex: Int; let order: Int; let staff: Staff }
    var tasks: [StaffTask] = []
    for (systemIndex, system) in systems.enumerated() {
        for (order, staff) in system.staffs.enumerated() {
            tasks.append(StaffTask(systemIndex: systemIndex, order: order, staff: staff))
        }
    }

    // Pre-sized output collector. Each task writes its own slot, so distinct
    // indices never collide; the lock only guards the (cheap) array store. Boxed
    // in a reference type so it can be shared across `@Sendable` work closures
    // without tripping strict-concurrency mutable-capture checks.
    let collector = StaffParseCollector(count: tasks.count)

    func runTask(_ i: Int) {
        let task = tasks[i]
        // The transformer (608-step decode) dominates; this is the parallel win.
        let symbols = parseStaffImage(
            index: i, staff: task.staff, image: image, regions: regions, config: config
        )
        // Labels only appear on the first system; skip OCR elsewhere.
        let label = task.systemIndex == 0
            ? InstrumentLabelReader.readLabel(image: labelImage, staff: task.staff)
            : nil
        collector.store(
            ParsedStaff(
                id: i,
                systemIndex: task.systemIndex,
                orderInSystem: task.order,
                // Per-page recognition is page-agnostic (always page 0 here); the
                // multi-page merge in `OMRProcessor` rewrites this to the real page.
                pageIndex: 0,
                minX: task.staff.minX, minY: task.staff.minY,
                maxX: task.staff.maxX, maxY: task.staff.maxY,
                label: label,
                symbols: symbols
            ),
            at: i
        )
    }

    // Decode staves in parallel. Each staff is independent and shares only
    // read-only inputs (image, regions) plus the thread-safe transformer
    // singleton. Width is capped to bound peak memory — every concurrent decode
    // holds its own KV-cache buffers.
    let width = min(maxConcurrentStaffParsing, tasks.count)
    if width <= 1 {
        for i in tasks.indices { runTask(i) }
    } else {
        let queue = DispatchQueue(label: "com.homr.ios.staff-parse", attributes: .concurrent)
        let group = DispatchGroup()
        let gate = DispatchSemaphore(value: width)
        for i in tasks.indices {
            gate.wait()
            queue.async(group: group) {
                runTask(i)
                gate.signal()
            }
        }
        group.wait()
    }

    return collector.ordered()
}

/// Thread-safe, fixed-size sink for parallel staff parsing. Writes go to a
/// pre-allocated slot per index (no resizing), serialised by a lock.
private final class StaffParseCollector: @unchecked Sendable {
    private var slots: [ParsedStaff?]
    private let lock = NSLock()

    init(count: Int) { slots = [ParsedStaff?](repeating: nil, count: count) }

    func store(_ staff: ParsedStaff, at index: Int) {
        lock.lock()
        slots[index] = staff
        lock.unlock()
    }

    /// Parsed staves in flattened (reading) order, dropping any empty slots.
    func ordered() -> [ParsedStaff] {
        lock.lock()
        defer { lock.unlock() }
        return slots.compactMap { $0 }
    }
}

/// Maximum number of staves decoded concurrently. Capped (rather than using the
/// full core count) because each in-flight transformer decode holds its own
/// KV-cache buffers, so this trades a little memory for wall-clock speed. Set to
/// `1` to fall back to fully sequential decoding.
private let maxConcurrentStaffParsing: Int = max(1, min(ProcessInfo.processInfo.activeProcessorCount, 4))
