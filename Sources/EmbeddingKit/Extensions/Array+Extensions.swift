public extension Array {
    func paddedOrTrimmed(to length: Int..., with value: Element) -> Self {
        paddedOrTrimmed(to: length, with: value)
    }

    private func paddedOrTrimmed(to length: [Int], with value: Element) -> Self {
        for length in length {
            if count < length {
                return self + Array(repeating: value, count: length - count)
            }
        }
        return Array(prefix(length.last!))
    }
}

public extension Array where Element: Numeric {
    func paddedOrTrimmed(to length: Int...) -> Self {
        paddedOrTrimmed(to: length, with: 0)
    }
}
