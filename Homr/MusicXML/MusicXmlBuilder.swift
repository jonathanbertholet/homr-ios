import Foundation

// MARK: - Minimal MusicXML element tree
//
// The Python `homr` generator builds its score with the `musicxml`
// package (`musicxml.xmlelement.xmlelement as mxl`), creating typed element
// objects (`mxl.XMLScorePartwise`, `mxl.XMLNote`, ...), wiring them with
// `.add_child(...)`, setting text with `value_=`/`._value`, and setting
// attributes via constructor keywords / `._set_attributes(...)`, then
// serialising with `.write(path)`.
//
// iOS has no such library, so `XMLNode` is a tiny generic element node that
// reproduces exactly that structure. Each Python `mxl.XMLFoo(...)` becomes an
// `XMLNode("foo-name", ...)` whose element name is the kebab-case MusicXML
// element that the `mxl` class maps to (e.g. `XMLScorePartwise` ->
// `score-partwise`, `XMLPartList` -> `part-list`, `XMLBeatType` ->
// `beat-type`). Children are kept in insertion order — the same order the
// Python `.add_child(...)` calls run in — so the emitted document mirrors the
// generator's call sequence.

/// A single MusicXML element: name, ordered attributes, optional text value and
/// ordered children. Reference type so the generator can hold and later mutate
/// nodes (e.g. `rebalance_measure_voices` rewrites `<voice>` values in place),
/// matching the mutable object model of the Python `mxl` elements.
final class XMLNode {
    /// The literal MusicXML element name, e.g. `"score-partwise"`, `"note"`.
    let name: String
    /// Attributes in insertion order. Kept as ordered pairs (not a dictionary)
    /// so serialised attribute order is stable and matches the Python kwargs
    /// order (e.g. `<ending type="start" number="1"/>`).
    var attributes: [(String, String)]
    /// Element text content, mirroring `mxl`'s `value_` / `_value`. Mutually
    /// exclusive with children in everything this generator produces.
    var value: String?
    /// Child elements in insertion order (the order of the `.add_child` calls).
    var children: [XMLNode]

    /// Create an element, optionally with text content (mirrors
    /// `mxl.XMLFoo(value_=...)` or a bare `mxl.XMLFoo()`).
    init(_ name: String, value: String? = nil) {
        self.name = name
        self.attributes = []
        self.value = value
        self.children = []
    }

    /// Convenience for the many `mxl.XMLFoo(value_=<int>)` call sites
    /// (fifths, octave, duration, line, alter, divisions, ...). The integer is
    /// rendered with `String(_:)`, matching Python's `str(int)`.
    convenience init(_ name: String, value: Int) {
        self.init(name, value: String(value))
    }

    /// Append a child and return it, mirroring `parent.add_child(child)` which
    /// returns the added child in the `mxl` API.
    @discardableResult
    func addChild(_ child: XMLNode) -> XMLNode {
        children.append(child)
        return child
    }

    /// Set (or replace) an attribute, mirroring the constructor keywords and
    /// `_set_attributes({...})` of `mxl`. Replacing in place preserves the
    /// original position when an attribute is set more than once.
    func setAttribute(_ name: String, _ value: String) {
        if let index = attributes.firstIndex(where: { $0.0 == name }) {
            attributes[index] = (name, value)
        } else {
            attributes.append((name, value))
        }
    }

    /// Integer overload for attributes such as `number=` / `tempo=` that Python
    /// passes as ints.
    func setAttribute(_ name: String, _ value: Int) {
        setAttribute(name, String(value))
    }

    /// Read an attribute value, mirroring `mxl` attribute access such as
    /// `barline.location`. Returns `nil` when unset.
    func attribute(_ name: String) -> String? {
        for (key, value) in attributes where key == name {
            return value
        }
        return nil
    }

    /// Direct children with the given element name. Reproduces
    /// `get_children_of_type(mxl.XMLFoo)` — element identity in `mxl` is the
    /// class, here it is the element name string.
    func children(named name: String) -> [XMLNode] {
        children.filter { $0.name == name }
    }

    // MARK: Serialisation

    /// Serialise this node (and descendants) as a full MusicXML 4.0 partwise
    /// document: the `<?xml ...?>` declaration plus the score-partwise DOCTYPE
    /// that MuseScore and other readers expect, followed by the element tree.
    ///
    /// Intended to be called on the `<score-partwise>` root. `indent` adds
    /// two-space indentation and newlines for human-readable output.
    func xmlString(indent: Bool = true) -> String {
        var out = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
        out +=
            "<!DOCTYPE score-partwise PUBLIC "
            + "\"-//Recordare//DTD MusicXML 4.0 Partwise//EN\" "
            + "\"http://www.musicxml.org/dtds/partwise.dtd\">\n"
        serialize(indent: indent, level: 0, into: &out)
        return out
    }

    /// Recursive element writer. An element with no children renders as either a
    /// self-closing tag (`<dot/>`) or an inline text element (`<step>C</step>`);
    /// an element with children renders as a block with indented children.
    private func serialize(indent: Bool, level: Int, into out: inout String) {
        let pad = indent ? String(repeating: "  ", count: level) : ""
        let newline = indent ? "\n" : ""

        out += pad + "<" + name
        for (key, value) in attributes {
            out += " " + key + "=\"" + XMLNode.escapeAttribute(value) + "\""
        }

        if children.isEmpty {
            if let value = value {
                out += ">" + XMLNode.escapeText(value) + "</" + name + ">" + newline
            } else {
                out += "/>" + newline
            }
            return
        }

        out += ">" + newline
        for child in children {
            child.serialize(indent: indent, level: level + 1, into: &out)
        }
        out += pad + "</" + name + ">" + newline
    }

    /// Escape the five XML text-sensitive characters. None of the generated
    /// MusicXML text values normally contain these, but the title is
    /// user/recognition supplied, so escaping keeps the document well-formed.
    private static func escapeText(_ string: String) -> String {
        var result = ""
        result.reserveCapacity(string.count)
        for character in string {
            switch character {
            case "&": result += "&amp;"
            case "<": result += "&lt;"
            case ">": result += "&gt;"
            default: result.append(character)
            }
        }
        return result
    }

    /// Escape attribute values (text escaping plus quotes).
    private static func escapeAttribute(_ string: String) -> String {
        var result = ""
        result.reserveCapacity(string.count)
        for character in string {
            switch character {
            case "&": result += "&amp;"
            case "<": result += "&lt;"
            case ">": result += "&gt;"
            case "\"": result += "&quot;"
            default: result.append(character)
            }
        }
        return result
    }
}

/// The generated MusicXML score, wrapping the `<score-partwise>` root.
///
/// Replaces the Python return value of `generate_xml(...)` (an `mxl.XMLElement`)
/// together with its `.write(path)` method.
struct MusicXmlDocument {
    /// The `<score-partwise>` root element.
    let root: XMLNode

    /// Serialised MusicXML 4.0 document string (declaration + DOCTYPE + tree).
    func xmlString() -> String {
        root.xmlString()
    }

    /// Write the serialised document to `url` as UTF-8, mirroring `xml.write(path)`.
    func write(to url: URL) throws {
        try xmlString().write(to: url, atomically: true, encoding: .utf8)
    }
}
