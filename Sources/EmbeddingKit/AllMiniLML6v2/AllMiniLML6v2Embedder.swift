import CoreML
import Foundation
import Tokenizers

public class AllMiniLML6v2Embedder: Embedder {
    // MARK: Lifecycle

    public init() async throws {
        tokenizer = try await AutoTokenizer.from(
            modelFolder: Bundle.module.resourceURL!.appending(path: "AllMiniLML6v2")
        )
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try await all_MiniLM_L6_v2.load(configuration: configuration)
    }

    // MARK: Public

    public let dimensions: Int = 384

    public func embed(_ text: String) async throws -> [Float] {
        let tokens = tokenizer.encode(text: text).paddedOrTrimmed(to: 512).map(Int32.init)
        let attentionMask: [Int32] = tokens.map { $0 == 0 ? 0 : 1 }

        let input = all_MiniLM_L6_v2Input(
            input_ids: MLMultiArray(MLShapedArray(scalars: tokens, shape: [1, 512])),
            attention_mask: MLMultiArray(MLShapedArray(scalars: attentionMask, shape: [1, 512]))
        )

        let output = try await model.prediction(input: input)

        return output.embeddingsShapedArray.scalars.map(Float.init)
    }

    // MARK: Private

    private var tokenizer: any Tokenizer
    private var model: all_MiniLM_L6_v2
}
