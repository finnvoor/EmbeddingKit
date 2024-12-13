@testable import EmbeddingKit
import Testing

@Test func testAllMiniLML6v2Embedder() async throws {
    let embedder: Embedder = try await AllMiniLML6v2Embedder()
    let embeddings = try await embedder.embed("Hello, world!")
    #expect(embeddings.count == embedder.dimensions)
}
