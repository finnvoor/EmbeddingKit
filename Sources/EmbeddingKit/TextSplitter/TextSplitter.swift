public protocol TextSplitter {
    func split(_ text: String) -> [String]
}
