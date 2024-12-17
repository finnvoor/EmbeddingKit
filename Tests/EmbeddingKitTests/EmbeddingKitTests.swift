@testable import EmbeddingKit
import Foundation
import Testing

@Test func testAllMiniLML6v2Embedder() async throws {
    let embedder: Embedder = try await AllMiniLML6v2Embedder()
    let embeddings = try await embedder.embed("Hello, world!")
    #expect(embeddings.count == embedder.dimensions)
}

@Test func testVectorStore() async throws {
    let embedder: Embedder = try await AllMiniLML6v2Embedder()
    let vectorStore = try VectorStore(dimensions: embedder.dimensions)

    let sentences = [
        "Developers are awesome!",
        "Developers are miserable"
    ]

    let embeddings = try await embedder.embed(sentences)
    for (index, embedding) in embeddings.enumerated() {
        try vectorStore.insert(embedding, id: index)
    }

    let queryEmbeddings = try await embedder.embed("Developers are bad")
    let nearest = try vectorStore.findNearest(queryEmbeddings)
    #expect(nearest == [1, 0])
}

@Test func testRetrieval() async throws {
    let (data, _) = try await URLSession.shared.data(
        from: URL(string: "https://en.wikipedia.org/wiki/List_of_common_misconceptions?action=raw")!
    )
    let text = String(decoding: data, as: UTF8.self)

    let textSplitter: TextSplitter = NLTextSplitter(unit: .paragraph)
    let embedder: Embedder = try await AllMiniLML6v2Embedder()
    let vectorStore = try VectorStore(dimensions: embedder.dimensions)

    let paragraphs = textSplitter.split(text)
    let embeddings = try await embedder.embed(paragraphs)
    for (index, embedding) in embeddings.enumerated() {
        try vectorStore.insert(embedding, id: index)
    }

    let queryEmbeddings = try await embedder.embed("Did cowboys wear cowboy hats?")
    let nearest = try vectorStore.findNearest(queryEmbeddings, limit: 1)
    print("--------------------")
    print(paragraphs[nearest[0]])
    print("--------------------")
}
