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

    public func embed(_ texts: [String]) async throws -> [[Float]] {
        struct Container: @unchecked Sendable {
            let tokenizer: any Tokenizer
            let model: all_MiniLM_L6_v2
        }
        let container = Container(tokenizer: tokenizer, model: model)

        var embeddings: [[Float]] = .init(repeating: [], count: texts.count)
        try await withThrowingTaskGroup(of: (Int, [Float]).self) { group in
            for (index, text) in texts.enumerated() {
                group.addTask {
                    try await (index, Self.embed(
                        text,
                        tokenizer: container.tokenizer,
                        model: container.model
                    ))
                }
            }
            for try await (index, embedding) in group {
                embeddings[index] = embedding
            }
        }
        return embeddings
    }

    // MARK: Private

    private var tokenizer: any Tokenizer
    private var model: all_MiniLM_L6_v2

    private static func embed(
        _ text: String,
        tokenizer: any Tokenizer,
        model: all_MiniLM_L6_v2
    ) async throws -> [Float] {
        let tokens = tokenizer.encode(text: text).paddedOrTrimmed(to: 512).map(Int32.init)
        let attentionMask: [Int32] = tokens.map { $0 == 0 ? 0 : 1 }

        let input = all_MiniLM_L6_v2Input(
            input_ids: MLShapedArray(scalars: tokens, shape: [1, 512]),
            attention_mask: MLShapedArray(scalars: attentionMask, shape: [1, 512])
        )

        let output = try await model.prediction(input: input)

        return output.embeddingsShapedArray.scalars.map(Float.init)
    }
}
