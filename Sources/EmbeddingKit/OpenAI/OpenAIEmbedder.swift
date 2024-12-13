import Foundation

public class OpenAIEmbedder: Embedder {
    // MARK: Lifecycle

    public init(model: Model, apiKey: String) {
        self.model = model
        self.apiKey = apiKey
    }

    // MARK: Public

    public enum Model {
        case textEmbedding3Large
        case textEmbedding3Small
        case textEmbedding3Ada002

        // MARK: Public

        public var name: String {
            switch self {
            case .textEmbedding3Large: "text-embedding-3-large"
            case .textEmbedding3Small: "text-embedding-3-small"
            case .textEmbedding3Ada002: "text-embedding-3-ada-002"
            }
        }

        public var dimensions: Int {
            switch self {
            case .textEmbedding3Large: 3072
            case .textEmbedding3Small: 1536
            case .textEmbedding3Ada002: 1536
            }
        }
    }

    public enum Error: Swift.Error {
        case malformedResponse
    }

    public let model: Model

    public var dimensions: Int { model.dimensions }

    public func embed(_ text: String) async throws -> [Float] {
        let url = URL(string: "https://api.openai.com/v1/embeddings")!
        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        struct Body: Encodable {
            let input: String
            let model: String
        }
        request.httpMethod = "POST"
        request.httpBody = try JSONEncoder().encode(Body(
            input: text,
            model: model.name
        ))
        // TODO: - check response
        let (data, _) = try await URLSession.shared.data(for: request)
        struct Response: Decodable {
            struct Item: Decodable {
                let embedding: [Float]
            }

            let data: [Item]
        }
        let response = try JSONDecoder().decode(Response.self, from: data)
        guard let embeddings = response.data.first?.embedding else { throw Error.malformedResponse }
        return embeddings
    }

    // MARK: Private

    private let apiKey: String
}
