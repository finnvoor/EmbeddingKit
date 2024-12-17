// MARK: - Embedder

public protocol Embedder {
    var dimensions: Int { get }

    func embed(_ text: String) async throws -> [Float]
    func embed(_ texts: [String]) async throws -> [[Float]]
}

public extension Embedder {
    func embed(_ text: String) async throws -> [Float] {
        try await embed([text])[0]
    }
}
