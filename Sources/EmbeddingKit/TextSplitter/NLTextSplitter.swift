import NaturalLanguage

public class NLTextSplitter: TextSplitter {
    // MARK: Lifecycle

    public init(unit: NLTokenUnit) {
        self.unit = unit
    }

    // MARK: Public

    public let unit: NLTokenUnit

    public func split(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: unit)
        tokenizer.string = text
        return tokenizer
            .tokens(for: text.startIndex..<text.endIndex)
            .map { String(text[$0]).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}
