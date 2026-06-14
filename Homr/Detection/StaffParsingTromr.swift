import Foundation

// MARK: - StaffParsingTromr
//
// Port of `homr/staff_parsing_tromr.py`.
//
// Runs the transformer OMR model on a single prepared staff image. Python keeps
// a module-global `inference: Staff2Score | None` that is lazily created on the
// first call and reused thereafter (the model is expensive to construct). We
// reproduce that with a private `static var` singleton, created with `try?`
// (Python's `Staff2Score(config)` can fail; here construction can throw, so a
// failed init leaves the singleton `nil` and yields an empty result).

/// Holder for the lazily-created, shared transformer instance (Python's global
/// `inference`). Kept private so the only entry points are the free functions
/// below.
///
/// `initLock` guards ONLY the one-time construction of the singleton, so two
/// staff threads racing on the first call can't build the model twice. Once
/// created, `predict` runs *without* this lock so `parseIndividualStaves` can
/// decode several staves in parallel. Concurrency safety is handled per-stage:
/// the CPU decoder session is safe to `Run` concurrently, while the CoreML
/// encoder serialises its own `Run` internally (see `Encoder.runLock`) because
/// ORT's CoreML EP is not concurrency-safe.
private enum TromrInference {
    /// The shared model instance, created on first use and reused afterward.
    static var shared: Staff2Score?
    /// Serializes construction of `shared` only (not inference).
    static let initLock = NSLock()

    /// Returns the shared instance, building it once on first call. A failed
    /// init (Python's `Staff2Score(config)` can fail) leaves it `nil`.
    static func instance(config: TransformerConfig) -> Staff2Score? {
        initLock.lock()
        defer { initLock.unlock() }
        if shared == nil {
            shared = try? Staff2Score(config: config)
        }
        return shared
    }
}

/// Port of `parse_staff_tromr`: thin wrapper around `predict_best`.
func parseStaffTromr(staff: Staff, staffImage: GrayscaleImage, config: TransformerConfig) -> [EncodedSymbol] {
    predictBest(staffImage, staff: staff, config: config)
}

/// Port of `predict_best`.
///
/// Lazily constructs the shared `Staff2Score` (guarded with `try?`), runs
/// `predict`, and — unless the staff is a grand staff — filters out any symbols
/// the model placed on the "lower" staff.
func predictBest(_ orgImage: GrayscaleImage, staff: Staff, config: TransformerConfig) -> [EncodedSymbol] {
    // Build (or fetch) the shared model under a one-time lock, then run inference
    // outside it so multiple staves can decode concurrently.
    guard let inference = TromrInference.instance(config: config) else {
        return []
    }
    let result = inference.predict(orgImage)
    if staff.isGrandstaff {
        return result
    }
    return result.filter { $0.position != "lower" }
}
