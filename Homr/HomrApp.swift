import SwiftUI

/// Entry point for the Homr iOS OMR app.
@main
struct HomrApp: App {
    @State private var modelStore = ModelStore()
    /// Persistent bookcase of scanned scores, restored from disk on launch.
    @State private var library = ScoreLibrary()

    var body: some Scene {
        WindowGroup {
            LibraryView()
                .environment(modelStore)
                .environment(library)
        }
    }
}
