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
    let vectorStore = try VectorStore<String>(dimensions: embedder.dimensions)

    let sentences = [
        "Developers are awesome!",
        "Developers are miserable"
    ]

    let embeddings = try await embedder.embed(sentences)
    for (embedding, sentence) in zip(embeddings, sentences) {
        try vectorStore.insert(embedding, metadata: sentence)
    }

    let queryEmbeddings = try await embedder.embed("Developers are bad")
    let nearest = try vectorStore.findNearest(queryEmbeddings)

    #expect(nearest[0].metadata == sentences[1])
    #expect(nearest[1].metadata == sentences[0])
}

@Test func testRetrieval() async throws {
    let (data, _) = try await URLSession.shared.data(
        from: URL(string: "https://en.wikipedia.org/wiki/List_of_common_misconceptions?action=raw")!
    )
    let text = String(decoding: data, as: UTF8.self)

    let textSplitter: TextSplitter = NLTextSplitter(unit: .paragraph)
    let embedder: Embedder = try await AllMiniLML6v2Embedder()
    let vectorStore = try VectorStore<String>(dimensions: embedder.dimensions)

    let paragraphs = textSplitter.split(text)
    let embeddings = try await embedder.embed(paragraphs)

    for (embedding, paragraph) in zip(embeddings, paragraphs) {
        try vectorStore.insert(embedding, metadata: paragraph)
    }

    let queryEmbeddings = try await embedder.embed("Did cowboys wear cowboy hats?")
    let nearest = try vectorStore.findNearest(queryEmbeddings, limit: 1).first!

    print("--------------------")
    print(nearest.metadata)
    print("--------------------")
}
