import NaturalLanguage

class NLTextSplitter: TextSplitter {
    // MARK: Lifecycle

    init(unit: NLTokenUnit) {
        self.unit = unit
    }

    // MARK: Internal

    let unit: NLTokenUnit

    func split(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: unit)
        tokenizer.string = text
        return tokenizer
            .tokens(for: text.startIndex..<text.endIndex)
            .map { String(text[$0]).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}
