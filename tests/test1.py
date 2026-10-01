from graphrag.embeddings.sentence_transformer import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder()

vector = embedder.embed("Alice works at Microsoft.")

print(type(vector))
print(vector.shape)
print(vector[:10])