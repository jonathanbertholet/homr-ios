// swift-tools-version:5.10
import PackageDescription

// Headless macOS validation harness for the homr OMR pipeline.
//
// This package compiles the exact same Foundation-only pipeline sources used by
// the iOS app (copied from ../Homr, minus the UIKit/SwiftUI files) plus a small
// `main.swift` that replicates `ImagePreprocessor` + `OMRProcessor` so we can run
// the port on the Mac and compare its MusicXML output against the Python
// reference. ONNX Runtime ships a macOS slice, so the same models run here.
let package = Package(
    name: "HomrValidate",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(
            url: "https://github.com/microsoft/onnxruntime-swift-package-manager",
            from: "1.24.1"
        ),
    ],
    targets: [
        .executableTarget(
            name: "HomrValidate",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "Sources/HomrValidate"
        ),
    ]
)
