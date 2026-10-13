import json
from pathlib import Path

import main


BASE_DIR = Path(__file__).parent
EVAL_DOCUMENTS_DIR = BASE_DIR / "evaluations" / "documents"
CASES_FILE = BASE_DIR / "evaluations" / "retrieval_cases.json"
TOP_K = 3


def load_cases() -> list[dict]:
    """读取检索评测问题。"""
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


def find_rank(results: list[dict], expected_keyword: str) -> int | None:
    """返回正确答案第一次出现的位置，从 1 开始计数。"""
    for index, result in enumerate(results, start=1):
        if expected_keyword in result["text"]:
            return index
    return None


def evaluate(name: str, cases: list[dict], retrieve_function) -> None:
    """运行一组检索问题并打印 Top-1、Top-3 命中率。"""
    top1_hits = 0
    top3_hits = 0

    print(f"\n=== {name} ===")

    for case in cases:
        query = case["query"]
        expected_keyword = case["expected_keyword"]
        results = retrieve_function(query)
        rank = find_rank(results, expected_keyword)

        if rank == 1:
            top1_hits += 1
        if rank is not None:
            top3_hits += 1

        if rank is None:
            status = "MISS"
        else:
            status = f"HIT@{rank}"

        best_result = results[0] if results else None
        if best_result is None:
            best_text = "没有检索结果"
            best_score = 0.0
        else:
            best_text = (
                f"{best_result['source']} #{best_result['chunk_index']}"
            )
            best_score = best_result["score"]

        print(f"[{status}] {query}")
        print(f"    期望关键词：{expected_keyword}")
        print(f"    Top-1：{best_text}，分数={best_score:.4f}")

    case_count = len(cases)
    print(f"Top-1 命中率：{top1_hits}/{case_count}")
    print(f"Top-3 命中率：{top3_hits}/{case_count}")


def run() -> None:
    cases = load_cases()
    chunks = main.load_chunks(EVAL_DOCUMENTS_DIR)

    print(f"评测文本块数量：{len(chunks)}")
    print(f"评测问题数量：{len(cases)}")
    print(f"Embedding 模型：{main.EMBEDDING_MODEL}")

    evaluate(
        "lexical",
        cases,
        lambda query: main.retrieve(query, chunks, top_k=TOP_K),
    )

    embedding_client = main.create_embedding_client()
    evaluate(
        "embedding",
        cases,
        lambda query: main.retrieve_with_embeddings(
            query,
            chunks,
            embedding_client,
            main.EMBEDDING_MODEL,
            top_k=TOP_K,
        ),
    )


if __name__ == "__main__":
    run()
