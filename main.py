from pathlib import Path


BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHUNK_SIZE = 120
CHUNK_OVERLAP = 20


def load_document(path: Path) -> str:
    """读取一个 UTF-8 文本文档。"""
    return path.read_text(encoding="utf-8")


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """把长文本切成小块，相邻块之间保留 overlap 个字符。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0，并且小于 chunk_size")

    normalized_text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(normalized_text):
        end = min(start + chunk_size, len(normalized_text))
        chunk = normalized_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(normalized_text):
            break

        start = end - overlap

    return chunks


def load_chunks() -> list[dict]:
    """读取 documents 目录下的文档，并为每块保存来源信息。"""
    all_chunks = []

    for path in sorted(DOCUMENTS_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = load_document(path)
        for index, chunk in enumerate(split_text(text)):
            all_chunks.append(
                {
                    "source": path.name,
                    "chunk_index": index,
                    "text": chunk,
                }
            )

    return all_chunks


def main():
    chunks = load_chunks()

    print(f"共读取到 {len(chunks)} 个文本块\n")

    for chunk in chunks:
        print(
            f"[{chunk['source']} #{chunk['chunk_index']}] "
            f"{chunk['text'][:80]}..."
        )


if __name__ == "__main__":
    main()
