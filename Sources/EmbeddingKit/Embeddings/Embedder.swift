// MARK: - Embedder

public protocol Embedder {
    var dimensions: Int { get }

    func embed(_ text: String) async throws -> [Float]
    func embed(_ texts: [String]) async throws -> [[Float]]
}

public extension Embedder {
    func embed(_ texts: [String]) async throws -> [[Float]] {
        var embeddings: [[Float]] = []
        for text in texts {
            try await embeddings.append(embed(text))
        }
        return embeddings
    }
}
