import EmbeddingKit
import USearch

public class USearchVectorStore: VectorStore {
    // MARK: Lifecycle

    public init(dimensions: Int) {
        index = USearchIndex.make(
            metric: .cos,
            dimensions: UInt32(dimensions),
            connectivity: 0,
            quantization: .F16
        )
    }

    // MARK: Public

    public func insert(_ embeddings: [[Float]]) throws -> [ID] {
        index.reserve(UInt32(embeddings.count))
        let nextIDs = (0..<embeddings.count).map { _ in nextID.next() }
        for (nextID, embedding) in zip(nextIDs, embeddings) {
            index.add(key: nextID, vector: embedding)
        }
        return nextIDs.map { Int($0) }
    }

    public func remove(_ id: ID) throws {
        index.remove(key: USearchKey(id))
    }

    public func findNearest(_ embedding: [Float], limit: Int) throws -> [(id: ID, distance: Double)] {
        let (ids, distances) = index.search(vector: embedding, count: limit)
        return zip(ids, distances).map { id, distance in
            (id: Int(id), distance: Double(distance))
        }
    }

    // MARK: Private

    private class NextID {
        // MARK: Internal

        func next() -> USearchKey {
            queue.sync {
                defer { nextID += 1 }
                return nextID
            }
        }

        // MARK: Private

        private var nextID: USearchKey = 0
        private let queue = DispatchQueue(label: "NextID")
    }

    private let index: USearchIndex
    private var nextID = NextID()
}
