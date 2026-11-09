"""Fixed-size character chunks. Later days compare this with other strategies."""

def fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be smaller than the chunk size")
    body = text.strip()
    if not body:
        return []
    step_floor = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(body):
        end = min(len(body), start + size)
        if end < len(body):
            snap = body.rfind(" ", start + step_floor, end)
            if snap > start:
                end = snap
        piece = body[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(body):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + step_floor
        start = next_start
    return chunks
