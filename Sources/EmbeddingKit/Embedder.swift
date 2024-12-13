public protocol Embedder {
    var dimensions: Int { get }

    func embed(_ text: String) async throws -> [Float]
}
