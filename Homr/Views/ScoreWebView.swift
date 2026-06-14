//
//  ScoreWebView.swift
//  Homr
//
//  A SwiftUI wrapper around WKWebView that renders a MusicXML string as real
//  music notation using the Verovio engraving toolkit, running fully OFFLINE.
//
//  How it works:
//   - The Verovio toolkit (verovio-toolkit-wasm.js) and a small index.html live
//     in the app bundle under the "Verovio" resource folder.
//   - We load index.html via `loadFileURL(_:allowingReadAccessTo:)` so the
//     WKWebView is allowed to read the sibling .js file from disk.
//   - The toolkit is fully self-contained (the WASM binary and the default music
//     fonts are embedded inside the .js as base64), so NO network access and no
//     separate .wasm/data files are required at runtime. This also sidesteps
//     WKWebView's lack of a "application/wasm" MIME mapping for file:// URLs,
//     since no standalone .wasm file is ever fetched.
//   - Once the page finishes loading we call the JS entry point
//     `window.renderMusicXML(xmlString)` with the MusicXML, JSON-encoded so the
//     string is safely escaped for JavaScript.
//

import SwiftUI
import WebKit

/// SwiftUI view that engraves and displays a MusicXML string offline.
///
/// Usage: `ScoreWebView(musicXML: someMusicXMLString)`
struct ScoreWebView: UIViewRepresentable {

    /// The MusicXML document to render. When this value changes, the view
    /// re-renders the score (see `updateUIView`).
    let musicXML: String

    /// Current playback position as a 0…1 fraction. When set, the notes sounding
    /// at that time are highlighted in the score (live playback follow). `nil`
    /// clears any highlight.
    var highlightFraction: Double? = nil

    // MARK: - Coordinator

    /// Bridges WKWebView navigation callbacks back into the representable.
    /// Acts as the `WKNavigationDelegate` so we know when index.html has
    /// finished loading and it is safe to inject JavaScript.
    final class Coordinator: NSObject, WKNavigationDelegate {

        /// The most recent MusicXML we were asked to render.
        var musicXML: String

        /// Becomes true after the initial page load finishes. Until then any
        /// render request is deferred (the JS side also queues, but we guard
        /// here too to avoid evaluating JS against a half-loaded page).
        var isLoaded = false

        /// The XML most recently handed to Verovio, so fraction-only updates
        /// don't trigger a (costly) full re-render of the score.
        var lastRenderedXML: String?

        init(musicXML: String) {
            self.musicXML = musicXML
        }

        /// Called by WebKit when index.html (and its scripts) finished loading.
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            isLoaded = true
            // Render the score we currently hold once the page is ready.
            render(in: webView, xml: musicXML)
        }

        /// Highlight the notes sounding at `fraction` (0…1), or clear if nil.
        func highlight(in webView: WKWebView, fraction: Double?) {
            guard let fraction else {
                webView.evaluateJavaScript("window.clearHighlight && window.clearHighlight();")
                return
            }
            webView.evaluateJavaScript("window.highlightAtFraction && window.highlightAtFraction(\(fraction));")
        }

        /// Surface load failures to the console for easier debugging.
        func webView(_ webView: WKWebView,
                     didFail navigation: WKNavigation!,
                     withError error: Error) {
            print("ScoreWebView: navigation failed -> \(error.localizedDescription)")
        }

        func webView(_ webView: WKWebView,
                     didFailProvisionalNavigation navigation: WKNavigation!,
                     withError error: Error) {
            print("ScoreWebView: provisional navigation failed -> \(error.localizedDescription)")
        }

        /// Safely injects the MusicXML into the page by calling the JS function
        /// `window.renderMusicXML(...)`. The XML is JSON-encoded so that any
        /// quotes, backslashes, newlines, or non-ASCII characters are correctly
        /// escaped into a single valid JavaScript string literal.
        func render(in webView: WKWebView, xml: String) {
            // JSON-encode the Swift string -> produces a quoted, escaped JS
            // string literal (e.g. "<score>\n...") that we can drop directly
            // into the JS call. Using JSONEncoder avoids manual escaping bugs.
            //
            // Note: this references the Foundation `JSONEncoder`. The app module
            // also declares its own `Encoder`/`Decoder` types, but those do not
            // collide here because we use the fully-qualified Foundation type
            // and never reference the bare app-level names.
            guard let data = try? JSONEncoder().encode(xml),
                  let jsLiteral = String(data: data, encoding: .utf8) else {
                print("ScoreWebView: failed to JSON-encode MusicXML")
                return
            }

            // Build the JS call, e.g. window.renderMusicXML("...escaped...");
            let js = "window.renderMusicXML(\(jsLiteral));"
            lastRenderedXML = xml   // mark before async eval to avoid double renders
            webView.evaluateJavaScript(js) { _, error in
                if let error = error {
                    print("ScoreWebView: render JS error -> \(error.localizedDescription)")
                }
            }
        }
    }

    /// Creates the coordinator that owns navigation state and the current XML.
    func makeCoordinator() -> Coordinator {
        Coordinator(musicXML: musicXML)
    }

    // MARK: - UIViewRepresentable

    /// Builds and configures the WKWebView, then loads the bundled index.html.
    func makeUIView(context: Context) -> WKWebView {
        // Default configuration is fine: the toolkit needs only JS + WASM,
        // both of which run in the standard web content process.
        let configuration = WKWebViewConfiguration()

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator

        // Allow the user to scroll and pinch-to-zoom the engraved score.
        webView.scrollView.isScrollEnabled = true
        webView.scrollView.bounces = true
        // The <meta viewport> in index.html enables user scaling; mirror that
        // intent on the native scroll view so pinch zoom feels natural.
        webView.scrollView.minimumZoomScale = 1.0
        webView.scrollView.maximumZoomScale = 5.0

        // White background avoids a black flash before content paints.
        webView.isOpaque = false
        webView.backgroundColor = .white
        webView.scrollView.backgroundColor = .white

        // Locate and load the bundled index.html (see helper below).
        if let indexURL = Self.bundledIndexURL() {
            // `allowingReadAccessTo` must grant access to the folder that holds
            // BOTH index.html and verovio-toolkit-wasm.js, otherwise the page
            // is blocked from reading its sibling script. We pass the directory.
            let readAccessURL = indexURL.deletingLastPathComponent()
            webView.loadFileURL(indexURL, allowingReadAccessTo: readAccessURL)
        } else {
            // Fail loudly in the console; the view will simply stay blank.
            print("ScoreWebView: could not locate Verovio index.html in bundle")
        }

        return webView
    }

    /// Called by SwiftUI when inputs change. Only re-renders when the MusicXML
    /// actually changed; otherwise just updates the live playback highlight (this
    /// is invoked frequently during playback as the position advances).
    func updateUIView(_ webView: WKWebView, context: Context) {
        context.coordinator.musicXML = musicXML
        guard context.coordinator.isLoaded else { return }
        if context.coordinator.lastRenderedXML != musicXML {
            context.coordinator.render(in: webView, xml: musicXML)
        }
        context.coordinator.highlight(in: webView, fraction: highlightFraction)
    }

    // MARK: - Bundle lookup

    /// Locates index.html inside the app bundle.
    ///
    /// xcodegen routes non-Swift files under `Homr/Resources/Verovio/` into the
    /// app's Copy Bundle Resources phase. Depending on how the folder is added,
    /// the resources may either:
    ///   (a) preserve the folder structure  -> ".../Verovio/index.html"
    ///   (b) be flattened to the bundle root -> ".../index.html"
    /// We try the subdirectory form first, then fall back to the flat form so
    /// the view keeps working regardless of how the bundle was assembled.
    static func bundledIndexURL() -> URL? {
        // (a) Folder structure preserved under "Verovio".
        if let url = Bundle.main.url(forResource: "index",
                                     withExtension: "html",
                                     subdirectory: "Verovio") {
            return url
        }
        // (b) Flattened into the bundle root.
        if let url = Bundle.main.url(forResource: "index",
                                     withExtension: "html") {
            return url
        }
        return nil
    }
}
