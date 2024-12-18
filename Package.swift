// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "EmbeddingKit",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [.library(name: "EmbeddingKit", targets: ["EmbeddingKit"])],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.14"),
        .package(url: "https://github.com/finnvoor/SQLiteVec.swift.git", branch: "main"),
        .package(url: "git@github.com:unum-cloud/usearch.git", from: "2.16.7")
    ],
    targets: [
        .target(name: "EmbeddingKit"),
        .target(
            name: "AllMiniLML6v2Embedder",
            dependencies: [
                "EmbeddingKit",
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            resources: [
                .copy("Resources/AllMiniLML6v2"),
            ]
        ),
        .target(
            name: "USearchVectorStore",
            dependencies: [
                "EmbeddingKit",
                .product(name: "USearch", package: "usearch")
            ]
        ),
        .target(
            name: "SQLiteVecVectorStore",
            dependencies: [
                "EmbeddingKit",
                .product(name: "SQLiteVec", package: "SQLiteVec.swift"),
            ]
        ),
        .testTarget(
            name: "EmbeddingKitTests",
            dependencies: [
                "EmbeddingKit",
                "AllMiniLML6v2Embedder",
                "SQLiteVecVectorStore",
                "USearchVectorStore"
            ]
        )
    ]
)
