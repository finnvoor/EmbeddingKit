import EmbeddingKit
import Foundation
import MLX
import MLXEmbedders

// MARK: - MLXAllMiniLML6v2Embedder

public class MLXAllMiniLML6v2Embedder: Embedder {
    // MARK: Lifecycle

    public init() async throws {
        MLX.GPU.set(memoryLimit: 100_000_000)
        modelContainer = try await MLXEmbedders.loadModelContainer(
            configuration: .minilm_l6
        )
    }

    // MARK: Public

    public let dimensions: Int = 384

    public func embed(_ texts: [String]) async throws -> [[Float]] {
        var embeddings: [[Float]] = []
        for text in texts {
            await embeddings.append(contentsOf: modelContainer.perform { model, tokenizer, pooling -> [[Float]] in
                let inputs = [tokenizer(text, addSpecialTokens: false)]
                // Pad to longest
                let maxLength = inputs.reduce(into: 16) { acc, elem in
                    acc = max(acc, elem.count)
                }

                let padded = stacked(
                    inputs.map { elem in
                        MLXArray(elem + Array(repeating: tokenizer.eosTokenId ?? 0, count: maxLength - elem.count))
                    }
                )
                let mask = (padded .!= tokenizer.eosTokenId ?? 0)
                let tokenTypes = MLXArray.zeros(like: padded)
                let result = pooling(
                    model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
                    normalize: true, applyLayerNorm: false
                )
                return result.map { $0.asArray(Float.self) }
            })
        }
        return embeddings
    }

    // MARK: Private

    private let modelContainer: ModelContainer
}
