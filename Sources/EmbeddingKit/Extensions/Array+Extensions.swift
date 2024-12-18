public extension Array {
    func paddedOrTrimmed(to length: Int, with value: Element) -> Self {
        if count >= length {
            Array(prefix(length))
        } else {
            self + Array(repeating: value, count: length - count)
        }
    }
}

public extension [Int] {
    func paddedOrTrimmed(to length: Int) -> Self {
        paddedOrTrimmed(to: length, with: 0)
    }
}
