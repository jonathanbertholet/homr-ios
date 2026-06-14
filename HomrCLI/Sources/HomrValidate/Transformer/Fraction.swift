import Foundation

/// Exact-arithmetic rational number, mirroring Python's `fractions.Fraction`.
///
/// `homr` relies on `Fraction` for note durations (e.g. `Fraction(1, 4)` for a
/// quarter note, dotted-note accumulation, and `Fraction(actual_notes,
/// normal_notes)` tuplet scaling). Swift has no built-in rational, so this is a
/// minimal exact value type.
///
/// Invariants (matching Python `Fraction`):
/// - Always stored in lowest terms (reduced by GCD on init).
/// - `denominator` is always strictly positive; the sign lives on `numerator`.
/// - `Fraction(actual, normal).numerator / .denominator` therefore reproduces
///   Python's `Fraction(actual_notes, normal_notes).numerator/.denominator`.
///
/// `Int` precision is sufficient for this domain (small kern durations).
struct Fraction: Comparable, Equatable, Hashable {
    /// Reduced numerator (carries the sign).
    let numerator: Int
    /// Reduced denominator, always > 0.
    let denominator: Int

    /// Whole-number fraction `n / 1` (mirrors Python `Fraction(n)`).
    init(_ n: Int) {
        self.numerator = n
        self.denominator = 1
    }

    /// Reduced fraction `n / d` (mirrors Python `Fraction(n, d)`).
    ///
    /// Normalizes the sign onto the numerator and divides both terms by their
    /// GCD, so `Fraction(2, 4) == Fraction(1, 2)` and `Fraction(-1, -2) ==
    /// Fraction(1, 2)`.
    init(_ n: Int, _ d: Int) {
        precondition(d != 0, "Fraction denominator must not be zero")
        var num = n
        var den = d
        // Keep the denominator positive; move any sign to the numerator.
        if den < 0 {
            num = -num
            den = -den
        }
        let g = Fraction.gcd(num, den)
        // g is >= 1 here (den != 0), so this always reduces to lowest terms.
        self.numerator = num / g
        self.denominator = den / g
    }

    /// Euclidean GCD on absolute values. `gcd(0, d) == d`, so `0 / d`
    /// reduces to `0 / 1`, matching Python.
    private static func gcd(_ a: Int, _ b: Int) -> Int {
        var x = abs(a)
        var y = abs(b)
        while y != 0 {
            (x, y) = (y, x % y)
        }
        return x
    }

    static func + (lhs: Fraction, rhs: Fraction) -> Fraction {
        Fraction(
            lhs.numerator * rhs.denominator + rhs.numerator * lhs.denominator,
            lhs.denominator * rhs.denominator
        )
    }

    static func - (lhs: Fraction, rhs: Fraction) -> Fraction {
        Fraction(
            lhs.numerator * rhs.denominator - rhs.numerator * lhs.denominator,
            lhs.denominator * rhs.denominator
        )
    }

    static func * (lhs: Fraction, rhs: Fraction) -> Fraction {
        Fraction(lhs.numerator * rhs.numerator, lhs.denominator * rhs.denominator)
    }

    static func / (lhs: Fraction, rhs: Fraction) -> Fraction {
        Fraction(lhs.numerator * rhs.denominator, lhs.denominator * rhs.numerator)
    }

    /// Cross-multiplied comparison. Both denominators are positive, so the
    /// inequality direction is preserved.
    static func < (lhs: Fraction, rhs: Fraction) -> Bool {
        lhs.numerator * rhs.denominator < rhs.numerator * lhs.denominator
    }
}
