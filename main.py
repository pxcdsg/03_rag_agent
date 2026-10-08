import math
from collections import Counter
from pathlib import Path
#上面是导包

BASE_DIR = Path(__file__).parent#定义本文件的源文件夹路径
DOCUMENTS_DIR = BASE_DIR / "documents"#定义源文件夹下的documents文件夹
CHUNK_SIZE = 120#不知道这个是定义什么，读后面发现这是预先定义一个文本块的大小规格
CHUNK_OVERLAP = 20#不知道这个是定义什么，读后面发现这是定义相邻文本文件的重合规格，是为了避免两个相邻文本文件内容出现割裂。


#定义一个读取文档的函数，我不知道这些参数是什么意思(path: Path) -> str:
def load_document(path: Path) -> str:
    """读取一个 UTF-8 文本文档。"""
    return path.read_text(encoding="utf-8")

#定义一个分割文本文件的函数，定义每个文本文件的长度为chunk_size然后定义相邻文本文件的重合规格，
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

    normalized_text = " ".join(text.split())#压缩多余空行和换行
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

#load_chunks() -> list[dict]:这个->是什么含义呀？然后这一部分是为每一个chunk保存了一个登记信息包括来源路径，自己的编号还有具体的内容
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

#这一部分是定义了一个字符转换的函数，但是我没有太看懂具体的转换细节，是不区分大小写了？还是什么其他的转换？
def tokenize(text: str) -> list[str]:
    """把文字转换成相邻字符特征，例如“遥感”转成“遥感”。"""
    chars = [char.lower() for char in text if char.isalnum()]

    if len(chars) < 2:
        return chars

    return ["".join(chars[index : index + 2]) for index in range(len(chars) - 1)]

#这个函数是用于计算每个字符在文本中出现的次数，但是我不太理解是每一个都记录吗？比如我：出现5次这样去记录的吗？那如果文本量大的时候这样子记录的效率是否会太低呢？
def build_term_frequency(text: str) -> Counter:
    """统计每个字符特征在文本中出现的次数。"""
    return Counter(tokenize(text))

#什么是词频向量？余弦相似度又是啥？这里是我的知识盲区了，我不太理解llm这个具体的内容以及rag这方面的知识。
def cosine_similarity(left: Counter, right: Counter) -> float:
    """计算两个词频向量的余弦相似度，结果在 0 到 1 之间。"""
    common_terms = left.keys() & right.keys()
    dot_product = sum(left[term] * right[term] for term in common_terms)

    left_length = math.sqrt(sum(value * value for value in left.values()))
    right_length = math.sqrt(sum(value * value for value in right.values()))

    if left_length == 0 or right_length == 0:
        return 0.0

    return dot_product / (left_length * right_length)

#打分又是什么操作？打分的标准是如何定义的？现在市面上统一使用一套打分标准吗？
def retrieve(question: str, chunks: list[dict], top_k: int = 2) -> list[dict]:
    """根据问题给每个文本块打分，返回得分最高的 top_k 块。"""
    question_vector = build_term_frequency(question)
    results = []

    for chunk in chunks:
        chunk_vector = build_term_frequency(chunk["text"])
        score = cosine_similarity(question_vector, chunk_vector)
        results.append({**chunk, "score": score})

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def main():
    chunks = load_chunks()

    if not chunks:
        print("documents 目录中没有找到 .md 或 .txt 文档。")
        return

    print(f"共读取到 {len(chunks)} 个文本块")
    question = input("请输入问题：").strip()

    if not question:
        print("问题不能为空。")
        return

    results = retrieve(question, chunks)
    print(f"\n最相关的 {len(results)} 个文本块：\n")

    for chunk in results:
        print(
            f"[{chunk['source']} #{chunk['chunk_index']}] "
            f"相似度={chunk['score']:.4f}"
        )
        print(f"{chunk['text']}\n")


if __name__ == "__main__":
    main()
