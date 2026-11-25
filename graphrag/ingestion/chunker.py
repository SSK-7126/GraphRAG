from graphrag.model.document import Chunk


def chunk_text(text: str,chunk_size: int = 500,overlap: int =50) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_document(
    text: str, document_id: str, *, chunk_size: int = 500, overlap: int = 50
) -> list[Chunk]:
    """Create stable, provenance-preserving chunks for the dataset pipeline."""
    return [
        Chunk(
            id=f"{document_id}:chunk:{position:05d}",
            text=chunk,
            document_id=document_id,
            position=position,
        )
        for position, chunk in enumerate(chunk_text(text, chunk_size, overlap), start=1)
    ]
