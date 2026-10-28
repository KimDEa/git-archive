import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
load_dotenv()


class Embedd:
    MODEL = "gemini-embedding-001"
    DIM = 768          # 벡터 길이 (숫자 몇 개로 표현할지)
    SAMPLE = 300       # 공부용으로 앞에서 300개만 사용 (전체는 시간이 오래 걸림)
    TOP_K = 5          # 결과를 몇 개 보여줄지
    MIN_SCORE = 0.5    # 이 점수보다 낮으면 관련 없다고 보고 버림

    def __init__(self):
        self.docs = [
            "A boy and his loyal dog go on a long journey home.",
            "A lonely girl adopts a stray puppy from the shelter.",
            "An old man and his canine companion survive the winter."
        ]
        self.netflex_csv_path = "/Users/d.h/dev/git-archive/personal/gemini_chatbot/netflix_titles.csv"
        self.folder = os.path.dirname(self.netflex_csv_path)
        self.embedding_path = os.path.join(self.folder, "vector_netflex.npy")
        self.api_key = os.environ["GEMINI_API_KEY"]

    def embed(self, task_type):
        client = genai.Client(api_key=self.api_key)

        vectors = []
        try:
            result = client.models.embed_content(
                model=Embedd.MODEL,
                contents=self.docs,
                config=types.EmbedContentConfig(
                    task_type=task_type,                  # 질문용인지 문서용인지
                    output_dimensionality=Embedd.DIM,
                ),
            )
        except Exception as e:
            print(f"error: {e}")
            return
        
        vectors += [e.values for e in result.embeddings]

        vectors = np.array(vectors)
        # 정규화: 각 벡터의 길이를 1로 맞춤 → 나중에 곱하기만 하면 코사인 유사도
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def vector_ready(self):
        if os.path.exists(self.embedding_path):
            doc_vectors = np.load(self.embedding_path)
            print("저장된 벡터를 불러옴.", doc_vectors.shape)
        else:
            print("벡터를 만듭니다.")
            doc_vectors = self.embed("RETRIEVAL_DOCUMENT")
            np.save(self.embedding_path, doc_vectors)
        print(f"저장 완료:, {doc_vectors.shape}")
        print(f"정규화 확인: {np.linalg.norm(doc_vectors, axis=1)}")
        print((doc_vectors @ doc_vectors.T).round(3))


if __name__ == "__main__":
    e = Embedd()
    e.vector_ready()        