from graphrag.embeddings.sentence_transformer import SentenceTransformerEmbedder


if __name__ == "__main__":
    embedder = SentenceTransformerEmbedder()
    vector = embedder.embed("Alice works at Microsoft.")
    print(type(vector))
    print(vector.shape)
    print(vector[:10])
