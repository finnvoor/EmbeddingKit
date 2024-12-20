import CoreML
import EmbeddingKit
import Foundation
import Tokenizers

// MARK: - CoreMLAllMiniLML6v2Embedder

public class CoreMLAllMiniLML6v2Embedder: Embedder {
    // MARK: Lifecycle

    public init() async throws {
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: Bundle.module.resourceURL!.appending(path: "AllMiniLML6v2")
        )
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let model = try await AllMiniLML6v2.load(configuration: configuration)
        let inputDimensions = model.model.modelDescription.inputDescriptionsByName[
            "input_ids"
        ]!.multiArrayConstraint!.shape[1].intValue
        modelContainer = ModelContainer(tokenizer: tokenizer, model: model, inputDimensions: inputDimensions)
        dimensions = model.model.modelDescription.outputDescriptionsByName[
            "embeddings"
        ]!.multiArrayConstraint!.shape[1].intValue
    }

    // MARK: Public

    public let dimensions: Int

    public func embed(_ texts: [String]) async throws -> [[Float]] {
        var embeddings: [[Float]] = .init(repeating: [], count: texts.count)
        try await withThrowingTaskGroup(of: (Int, [Float]).self) { [modelContainer] group in
            for (index, text) in texts.enumerated() {
                group.addTask {
                    let tokens = modelContainer.tokenizer.encode(text: text)
                        .map(Int32.init)
                        .paddedOrTrimmed(to: modelContainer.inputDimensions)

                    let input = AllMiniLML6v2Input(
                        input_ids: MLShapedArray(
                            scalars: tokens,
                            shape: [1, tokens.count]
                        )
                    )

                    let output = try await modelContainer.model.prediction(input: input)

                    return (index, output.embeddingsShapedArray.scalars.map(Float.init))
                }
            }
            for try await (index, embedding) in group {
                embeddings[index] = embedding
            }
        }
        return embeddings
    }

    // MARK: Private

    private let modelContainer: ModelContainer
}

// MARK: CoreMLAllMiniLML6v2Embedder.ModelContainer

extension CoreMLAllMiniLML6v2Embedder {
    private struct ModelContainer: @unchecked Sendable {
        let tokenizer: any Tokenizer
        let model: AllMiniLML6v2
        let inputDimensions: Int
    }
}
