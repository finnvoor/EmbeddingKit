@testable import AllMiniLML6v2Embedder
@testable import EmbeddingKit
import Foundation
@testable import SQLiteVecVectorStore
import Testing
@testable import USearchVectorStore

@Test func testAllMiniLML6v2Embedder() async throws {
    let embedder: Embedder = try await AllMiniLML6v2Embedder()
    let embeddings = try await embedder.embed("Hello, world!")
    #expect(embeddings.count == embedder.dimensions)
}

@Test func testVectorStore() async throws {
    let embedder: Embedder = try await AllMiniLML6v2Embedder()
    for vectorStore: VectorStore in try [
        SQLiteVecVectorStore(dimensions: embedder.dimensions),
        USearchVectorStore(dimensions: embedder.dimensions)
    ] {
        let sentences = [
            "Developers are awesome!",
            "Developers are miserable"
        ]

        let embeddings = try await embedder.embed(sentences)
        let ids = try vectorStore.insert(embeddings)
        let mapping = Dictionary(uniqueKeysWithValues: zip(ids, sentences))

        let queryEmbeddings = try await embedder.embed("Developers are bad")
        let nearest = try vectorStore.findNearest(queryEmbeddings, limit: 2)

        #expect(mapping[nearest[0].id] == sentences[1])
        #expect(mapping[nearest[1].id] == sentences[0])
    }
}

@Test func testRetrieval() async throws {
    let (data, _) = try await URLSession.shared.data(
        from: URL(string: "https://en.wikipedia.org/wiki/List_of_common_misconceptions?action=raw")!
    )
    let text = String(decoding: data, as: UTF8.self)

    let textSplitter: TextSplitter = NLTextSplitter(unit: .paragraph)
    let embedder: Embedder = try await AllMiniLML6v2Embedder()
    let vectorStore: VectorStore = try SQLiteVecVectorStore(dimensions: embedder.dimensions)

    let paragraphs = textSplitter.split(text)
    let embeddings = try await embedder.embed(paragraphs)
    let ids = try vectorStore.insert(embeddings)
    let mapping = Dictionary(uniqueKeysWithValues: zip(ids, paragraphs))

    let queryEmbeddings = try await embedder.embed("Did cowboys wear cowboy hats?")
    let nearest = try vectorStore.findNearest(queryEmbeddings, limit: 1)[0]

    print("--------------------")
    print(mapping[nearest.id]!)
    print("--------------------")
}
