// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "EmbeddingKit",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [.library(name: "EmbeddingKit", targets: ["EmbeddingKit"])],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.14"),
        .package(url: "https://github.com/finnvoor/SQLiteVec.swift.git", branch: "main")
    ],
    targets: [
        .target(
            name: "EmbeddingKit",
            dependencies: [
                .product(name: "SQLiteVec", package: "SQLiteVec.swift"),
                .product(name: "Transformers", package: "swift-transformers")
            ],
            resources: [
                .copy("Resources/AllMiniLML6v2"),
            ]
        ),
        .testTarget(
            name: "EmbeddingKitTests",
            dependencies: ["EmbeddingKit"],
            resources: [.process("Resources")]
        )
    ]
)
