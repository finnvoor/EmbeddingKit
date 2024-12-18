import Foundation

// MARK: - VectorStore

public protocol VectorStore {
    typealias ID = Int

    @discardableResult func insert(
        _ embedding: [Float]
    ) throws -> ID

    @discardableResult func insert(
        _ embeddings: [[Float]]
    ) throws -> [ID]

    func remove(_ id: ID) throws

    func findNearest(_ embedding: [Float], limit: Int) throws -> [(id: ID, distance: Double)]
}

public extension VectorStore {
    @discardableResult func insert(
        _ embedding: [Float]
    ) throws -> ID {
        try insert([embedding])[0]
    }
}
