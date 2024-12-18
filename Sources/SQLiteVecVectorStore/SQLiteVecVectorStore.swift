import EmbeddingKit
import SQLiteVec

public class SQLiteVecVectorStore: VectorStore {
    // MARK: Lifecycle

    public init(location: Connection.Location = .inMemory, dimensions: Int) throws {
        connection = try Connection(location)

        try connection.prepare(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
              embedding float[\(dimensions)]
            );
            """
        ).run()
    }

    // MARK: Public

    public func insert(_ embeddings: [[Float]]) throws -> [ID] {
        let statement = try connection.prepare("""
            INSERT INTO embeddings(embedding)
            VALUES (?)
        """)
        return try embeddings.map { embedding in
            try statement.run(embedding)
            return Int(connection.lastInsertRowid)
        }
    }

    public func remove(_ id: ID) throws {
        try connection.prepare("DELETE FROM embeddings WHERE rowid = ?").run(id)
    }

    public func findNearest(_ embedding: [Float], limit: Int = 20) throws -> [(id: ID, distance: Double)] {
        try connection.prepare("""
            SELECT rowid, distance
            FROM embeddings
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """).run(embedding, limit).map { row in
            (
                id: Int(row[0] as! Int64),
                distance: row[1] as! Double
            )
        }
    }

    // MARK: Private

    private let connection: Connection
}
