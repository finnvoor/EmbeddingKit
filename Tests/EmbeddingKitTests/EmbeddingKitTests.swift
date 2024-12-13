@testable import EmbeddingKit
import Testing

@Test func testAllMiniLML6v2Embedder() async throws {
    let embedder: Embedder = try await AllMiniLML6v2Embedder()
    let embeddings = try await embedder.embed("Hello, world!")
    #expect(embeddings.count == embedder.dimensions)
}

@Test func testVectorStore() async throws {
    let vectorStore = try VectorStore(dimensions: 384)
    let embedder: Embedder = try await AllMiniLML6v2Embedder()

    let sentences = [
        "Developers are awesome!",
        "Developers are miserable"
    ]
    for (index, sentence) in sentences.enumerated() {
        let embeddings = try await embedder.embed(sentence)
        try vectorStore.insert(embeddings, id: index)
    }

    let queryEmbeddings = try await embedder.embed("Developers are bad")
    let nearest = try vectorStore.findNearest(queryEmbeddings)
    #expect(nearest == [1, 0])
}
