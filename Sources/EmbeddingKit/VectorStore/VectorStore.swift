import Foundation
import SQLiteVec

// MARK: - VectorStore

public class VectorStore<Metadata: Codable> {
    // MARK: Lifecycle

    public init(location: Connection.Location = .inMemory, dimensions: Int) throws {
        connection = try Connection(location)

        try connection.prepare(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
              embedding float[\(dimensions)],
              +metadata blob
            );
            """
        ).run()
    }

    // MARK: Public

    public typealias ID = Int64

    @discardableResult public func insert(
        _ embedding: [Float],
        metadata: Metadata
    ) throws -> ID {
        let insert = try connection.prepare("""
            INSERT INTO embeddings(embedding, metadata)
            VALUES (?, ?)
        """)
        var metadataData = try JSONEncoder().encode(metadata)
        let count = metadataData.count
        try metadataData.withUnsafeMutableBytes { p in
            let blob = Blob(bytes: p.baseAddress!, length: count)
            try insert.run(embedding, blob)
        }
        return connection.lastInsertRowid
    }

    public func findNearest(_ embedding: [Float], limit: Int = 20) throws -> [Result] {
        try connection.prepare("""
            SELECT rowid, distance, metadata
            FROM embeddings
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """).run(embedding, limit).map { row in
            try Result(
                id: row[0] as! Int64,
                distance: row[1] as! Double,
                metadata: JSONDecoder().decode(
                    Metadata.self,
                    from: Data((row[2] as! Blob).bytes)
                )
            )
        }
    }

    // MARK: Private

    private let connection: Connection
}

// MARK: VectorStore.Result

public extension VectorStore {
    struct Result {
        public let id: ID
        public let distance: Double
        public let metadata: Metadata
    }
}
