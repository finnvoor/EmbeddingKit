import Foundation
import SQLiteVec

public class VectorStore {
    // MARK: Lifecycle

    public init(location: Connection.Location = .inMemory, dimensions: Int) throws {
        connection = try Connection(location)

        try connection.prepare(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
              id INTEGER PRIMARY KEY,
              embedding float[\(dimensions)]
            );
            """
        ).run()
    }

    // MARK: Public

    public func insert(_ embedding: [Float], id: Int) throws {
        let insert = try connection.prepare("""
            INSERT INTO embeddings(id, embedding)
            VALUES (?, ?)
        """)
        try insert.run(id, embedding)
    }

    public func findNearest(_ embedding: [Float], limit: Int = 20) throws -> [Int] {
        try connection.prepare("""
            SELECT id, distance
            FROM embeddings
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """).run(embedding, limit).map { $0[0] as! Int64 }.map(Int.init)
    }

    // MARK: Private

    private let connection: Connection
}
