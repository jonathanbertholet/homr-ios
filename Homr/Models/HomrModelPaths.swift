import Foundation

/// Paths and URLs for homr ONNX checkpoints (mirrors homr Python config).
enum HomrModelPaths {
    // Segmentation model (UNet) — same hashes as homr/segmentation/config.py
    static let segnetBaseName = "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f"
    static let transformerBaseName = "pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903"

    static let downloadBaseURL = URL(
        string: "https://github.com/liebharc/homr/releases/download/onnx_checkpoints/"
    )!

    /// Models required on iOS: fp32 segnet + fp32 encoder/decoder (CoreML EP cannot run decoder).
    static let requiredModels: [ModelDescriptor] = [
        ModelDescriptor(
            fileName: "\(segnetBaseName).onnx",
            zipName: "\(segnetBaseName).zip"
        ),
        ModelDescriptor(
            fileName: "encoder_\(transformerBaseName).onnx",
            zipName: "encoder_\(transformerBaseName).zip"
        ),
        ModelDescriptor(
            fileName: "decoder_\(transformerBaseName).onnx",
            zipName: "decoder_\(transformerBaseName).zip"
        ),
    ]

    static func modelsDirectory() -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return appSupport.appendingPathComponent("Models", isDirectory: true)
    }

    static func path(for fileName: String) -> URL {
        modelsDirectory().appendingPathComponent(fileName)
    }
}

struct ModelDescriptor: Identifiable, Hashable {
    let fileName: String
    let zipName: String

    var id: String { fileName }

    var downloadURL: URL {
        HomrModelPaths.downloadBaseURL.appendingPathComponent(zipName)
    }

    var localURL: URL {
        HomrModelPaths.path(for: fileName)
    }
}
