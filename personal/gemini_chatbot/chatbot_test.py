"""줄거리(description) 임베딩 검색 - 공부용 간단 버전

흐름
  [준비] 줄거리들 → 벡터로 바꾸기 → 파일로 저장 (처음 한 번만)
  [검색] 질문 → 벡터로 바꾸기 → 줄거리 벡터들과 유사도 계산 → 점수 높은 순으로 보여주기
"""
import os
import time
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

CSV_PATH = "/Users/d.h/dev/git-archive/personal/gemini_chatbot/netflix_titles.csv"
VECTOR_PATH = "/Users/d.h/dev/git-archive/personal/gemini_chatbot/study_vectors.npy"

MODEL = "gemini-embedding-001"
DIM = 768          # 벡터 길이 (숫자 몇 개로 표현할지)
SAMPLE = 300       # 공부용으로 앞에서 300개만 사용 (전체는 시간이 오래 걸림)
TOP_K = 5          # 결과를 몇 개 보여줄지
MIN_SCORE = 0.5    # 이 점수보다 낮으면 관련 없다고 보고 버림

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# ─── 1단계: 문장들을 벡터로 바꾸는 함수 ─────────────────────
def embed(texts, task_type):
    vectors = []
    for i in range(0, len(texts), 100):              # 100개씩 나눠서 요청
        batch = texts[i:i + 100]
        result = client.models.embed_content(
            model=MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,                  # 질문용인지 문서용인지
                output_dimensionality=DIM,
            ),
        )
        vectors += [e.values for e in result.embeddings]
        print(f"  임베딩 {len(vectors)}/{len(texts)}")

        if i + 100 < len(texts):
            time.sleep(61)                            # 무료 한도: 1분에 100개

    vectors = np.array(vectors)
    # 정규화: 각 벡터의 길이를 1로 맞춤 → 나중에 곱하기만 하면 코사인 유사도
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# ─── 2단계: 데이터 읽고, 줄거리 벡터 준비하기 ───────────────
df = pd.read_csv(CSV_PATH, index_col=0).fillna("Unknown").head(SAMPLE)

if os.path.exists(VECTOR_PATH):                       # 전에 저장해 둔 파일이 있으면
    doc_vectors = np.load(VECTOR_PATH)                # 그걸 읽어오고
    print("저장된 벡터 불러옴:", doc_vectors.shape)
else:                                                 # 없으면 새로 만들어서 저장
    print("줄거리 벡터를 처음 만듭니다.")
    doc_vectors = embed(df["description"].tolist(), "RETRIEVAL_DOCUMENT")
    np.save(VECTOR_PATH, doc_vectors)
    print("저장 완료:", doc_vectors.shape)            # (300, 768) = 300개 작품 × 숫자 768개


# ─── 3단계: 검색하는 함수 ────────────────────────────────
def search(question):
    query_vector = embed([question], "RETRIEVAL_QUERY")[0]   # 질문도 벡터로

    scores = doc_vectors @ query_vector               # 300개 줄거리와의 유사도를 한 번에 계산

    result = df.copy()
    result["score"] = scores.round(3)                 # 작품마다 점수 붙이기
    result = result[result["score"] >= MIN_SCORE]     # 너무 낮은 건 버리기
    return result.sort_values("score", ascending=False).head(TOP_K)


# ─── 4단계: 직접 검색해 보기 ─────────────────────────────
if __name__ == "__main__":
    while True:
        question = input("\n검색어 (q: 종료): ")
        if question == "q":
            break
        found = search(question)
        if found.empty:
            print("관련 작품이 없어요. MIN_SCORE를 낮춰보세요.")
        for _, row in found.iterrows():
            print(f"[{row['score']}] {row['title']}\n    {row['description']}")