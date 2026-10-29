import os
import numpy as np
import ollama

MODEL = "bge-m3"
BATCH = 100


def embed(texts):
    """문장 리스트 → 정규화된 벡터 배열 (문장 수, 1024)"""
    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        result = ollama.embed(model=MODEL, input=batch)
        vectors.extend(result["embeddings"])
        if len(texts) > BATCH:
            print(f"임베딩 {len(vectors)}/{len(texts)}")
    vectors = np.array(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def load_or_build(docs, path):
    """저장된 벡터가 있고 개수가 맞으면 불러오고, 아니면 만들어서 저장"""
    if os.path.exists(path):
        saved = np.load(path)
        if len(saved) == len(docs):
            print("저장된 벡터 불러옴:", saved.shape)
            return saved
        print("문서 수가 달라서 다시 만듭니다.")
    vectors = embed(docs)
    np.save(path, vectors)
    print("저장 완료:", vectors.shape)
    return vectors


def search(query, docs, doc_vectors, top_k=5):
    """질문과 가장 비슷한 문서 top_k개"""
    q = embed([query])[0]
    scores = doc_vectors @ q
    order = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), docs[i]) for i in order]


if __name__ == "__main__":
    docs = [
        "A boy and his loyal dog go on a long journey home.",
        "A lonely girl adopts a stray puppy from the shelter.",
        "Astronauts fight a hostile alien species in deep space.",
    ]
    doc_vectors = load_or_build(docs, f"vectors_{MODEL}.npy")

    for score, doc in search("강아지가 나오는 이야기", docs, doc_vectors):
        print(f"{score:.3f}  {doc}")