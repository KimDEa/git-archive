from log_config import logger
from google import genai
from google.genai import types
import os
import time
import json
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────
# 임베딩 설정
# ─────────────────────────────────────────────
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768          # 벡터 길이. 3072(기본)보다 작게 해서 저장 공간·계산을 줄인다
BATCH_SIZE = 100         # 한 번의 API 요청에 보낼 문장 수
# 무료 등급은 1분에 문장 100개까지라서, 묶음 하나 보낼 때마다 이만큼 쉰다
WAIT_BETWEEN_BATCHES = 61

# 최종 점수 = 의미 점수 × SEMANTIC_WEIGHT + 키워드 점수 × KEYWORD_WEIGHT
SEMANTIC_WEIGHT = 0.8
KEYWORD_WEIGHT = 0.2
TOP_K = 10               # 최종적으로 보여줄 작품 수
MIN_SCORE = 0.5          # 이 점수보다 낮으면 관련 없다고 보고 버린다


class SearchQuery(BaseModel):
    type: str | None = Field(
        description='"Movie" 또는 "TV Show". 언급이 없으면 null'
    )
    title: list[str] = Field(
        description='영화제목. 언급이 없으면 빈 리스트'
    )
    director: list[str] = Field(
        description='감독이름. 언급이 없으면 빈 리스트'
    )
    cast: list[str] = Field(
        description='출연진 이름. 여러개일수 있는데 언급이 없으면 빈 리스트'
    )
    country: list[str] = Field(
        description='만든 국가명. 언급이 없으면 빈 리스트'
    )
    listed_in: list[str] = Field(
        description="장르 목록. 언급이 없으면 빈 리스트"
    )
    year_from: int | None = Field(
        description="시작 연도. 예: 2000's → 2000, 'from 2000' → 2000"
    )
    year_to: int | None = Field(
        description="끝 연도. 예: 2000's → 2009, 'to 2012' → 2012"
    )
    release_year: int | None = Field(
        description="연도 하나만 언급되면 int형으로 저장. 언급이 없으면 null"
    )
    description: list[str] = Field(
        description="프로그램 줄거리. 줄거리에 포함될거 같은 단어를 문자열로 리스트에 저장. 언급이 없으면 빈 리스트"
    )


class Ai:
    def __init__(self):
        self.netflex_csv_path = "/Users/d.h/dev/git-archive/personal/gemini_chatbot/netflix_titles.csv"
        self.catagory_json_path = "/Users/d.h/dev/git-archive/personal/gemini_chatbot/catagory.json"
        # 임베딩 결과를 저장해 둘 파일 (CSV와 같은 폴더)
        self.embedding_path = os.path.join(
            os.path.dirname(self.netflex_csv_path), "description_embeddings.npy"
        )
        self.api_key = os.environ["GEMINI_API_KEY"]
        self.client = genai.Client(api_key=self.api_key)

        self.df = pd.read_csv(self.netflex_csv_path, index_col=0)
        self.df = self.df.fillna('Unknown')

        self.genres = sorted(
            self.df['listed_in'].str.split(',').explode().str.strip().unique()
            )

        # description 임베딩 불러오기 (없으면 처음 한 번 만든다)
        self.desc_vectors = self.load_or_build_embeddings()

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)

    # ─────────────────────────────────────────
    # 임베딩 관련
    # ─────────────────────────────────────────
    def embed(self, texts, task_type, partial_path=None):
        """문장 리스트를 벡터(숫자 리스트)들로 바꾼다. 결과 모양: (문장 수, EMBED_DIM)

        partial_path를 주면 묶음마다 중간 결과를 저장해서,
        중간에 멈춰도 다음 실행 때 이어서 만든다.
        """
        vectors = []
        if partial_path and os.path.exists(partial_path):
            vectors = np.load(partial_path).tolist()
            logger.info(f"이전에 만들던 임베딩 이어서 진행: {len(vectors)}/{len(texts)}")

        for start in range(len(vectors), len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]

            for attempt in range(5):  # 요청 한도에 걸리면 쉬고 다시 시도
                try:
                    result = self.client.models.embed_content(
                        model=EMBED_MODEL,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=EMBED_DIM,
                        ),
                    )
                    break
                except Exception as e:
                    if "429" not in str(e):  # 한도 문제가 아니면 기다려도 소용없으니 바로 중단
                        raise
                    logger.warning(f"요청 한도 초과, {WAIT_BETWEEN_BATCHES}초 쉬고 다시 시도 ({attempt + 1}/5)")
                    time.sleep(WAIT_BETWEEN_BATCHES)
            else:
                raise RuntimeError("임베딩 요청이 계속 실패했습니다. 일일 한도를 다 썼을 수 있어요.")

            vectors.extend(e.values for e in result.embeddings)

            if len(texts) > BATCH_SIZE:  # 여러 묶음을 보내는 경우 (처음 전체 임베딩)
                done = len(vectors)
                left_min = (len(texts) - done) // BATCH_SIZE * WAIT_BETWEEN_BATCHES // 60
                logger.info(f"임베딩 진행: {done}/{len(texts)} (남은 시간 약 {left_min}분)")
                if partial_path:
                    np.save(partial_path, np.array(vectors, dtype=np.float32))
                if done < len(texts):
                    time.sleep(WAIT_BETWEEN_BATCHES)

        vectors = np.array(vectors, dtype=np.float32)
        # 길이를 1로 맞춘다(정규화). 그러면 내적 = 코사인 유사도가 된다
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    def load_or_build_embeddings(self):
        """저장된 임베딩이 있으면 불러오고, 없으면 전체 description을 임베딩해서 저장한다."""
        if os.path.exists(self.embedding_path):
            vectors = np.load(self.embedding_path)
            if len(vectors) == len(self.df):
                logger.info(f"저장된 임베딩 불러옴: {vectors.shape}")
                return vectors
            logger.warning("CSV 행 수와 임베딩 수가 달라서 다시 만듭니다.")

        logger.info("description 임베딩을 처음 만듭니다. 몇 분 걸릴 수 있어요.")
        texts = self.df['description'].astype(str).tolist()
        partial_path = self.embedding_path.replace(".npy", "_partial.npy")
        vectors = self.embed(texts, task_type="RETRIEVAL_DOCUMENT", partial_path=partial_path)
        np.save(self.embedding_path, vectors)
        if os.path.exists(partial_path):
            os.remove(partial_path)  # 다 만들었으니 중간 저장 파일은 지운다
        logger.info(f"임베딩 저장 완료: {self.embedding_path}")
        return vectors

    def rank_by_description(self, result, keywords, user_input):
        """필터링된 작품들을 description 의미 유사도 + 키워드 일치로 점수를 매겨 정렬한다."""
        if result.empty:
            return result

        # 1) 질문을 벡터로 바꾼다. 키워드만 쓰는 것보다 원래 문장을 같이 넣으면 의미가 더 잘 잡힌다
        query_text = f"{user_input} / keywords: {', '.join(keywords)}"
        query_vec = self.embed([query_text], task_type="RETRIEVAL_QUERY")[0]

        # 2) 필터링 후 남은 작품들의 벡터만 꺼낸다 (df에서 몇 번째 행인지로 찾음)
        positions = self.df.index.get_indexer(result.index)
        doc_vecs = self.desc_vectors[positions]

        # 3) 의미 점수: 코사인 유사도 (정규화했으니 내적만 하면 된다)
        semantic = doc_vecs @ query_vec

        # 4) 키워드 점수: 키워드 중 몇 개가 description에 그대로 들어 있는지 비율
        desc_lower = result['description'].str.lower()
        hits = sum(desc_lower.str.contains(k.lower(), regex=False) for k in keywords)
        keyword = np.asarray(hits, dtype=np.float32) / len(keywords)

        # 5) 가중치를 곱해서 합친 최종 점수
        result = result.copy()
        result['semantic_score'] = semantic.round(3)
        result['keyword_score'] = keyword.round(3)
        result['score'] = (SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * keyword).round(3)

        result = result[result['score'] >= MIN_SCORE]
        return result.sort_values('score', ascending=False).head(TOP_K)

    # ─────────────────────────────────────────
    # 검색 흐름
    # ─────────────────────────────────────────
    def extract_keyword(self, user_input):
        client = self.client

        prompt = (
            f"""
            너는 {user_input}으로 받은 단어를 가지고 분류를 할거야.
            기본적으로 자연어스러운 입력으로 부터 키워드를 분류해서 알맞은 넷플릭스 프로그램을 찾아주는일을 하는거야.
            데이터는 모두 영어로 되어있으니 한글로 받아도 영어로 바꿔줘.
            {self.genres}안에 포함된것으로만 장르로 구별하고 포함이 안되면 설명이나 프로그램 이름에 포함될수도 있어."""
        )

        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                    response_schema=SearchQuery,
                ),
            )
            keyword_result = json.loads(response.text)

        except Exception:
            logger.exception('error')
            return

        else:
            with open(self.catagory_json_path, mode='w', encoding='utf-8') as f:
                json.dump(keyword_result, f, indent=4, ensure_ascii=False)

        if not keyword_result:
            print("File is Empty.")
            return

        def extract_input(keyword_result):
            result = self.df.copy()

            if keyword_result['type']:
                result = result[result['type'] == keyword_result['type']]
            if keyword_result['title']:
                t = "|".join(keyword_result['title'])
                result = result[result['title'].str.contains(t, na=False, case=False)]
            if keyword_result['director']:
                d = "|".join(keyword_result['director'])
                result = result[result['director'].str.contains(d, na=False, case=False)]
            if keyword_result['cast']:
                c = "|".join(keyword_result['cast'])
                result = result[result['cast'].str.contains(c, na=False, case=False)]
            if keyword_result['country']:
                c = "|".join(keyword_result['country'])
                result = result[result['country'].str.contains(c, na=False, case=False)]
            if keyword_result['listed_in']:
                l = "|".join(keyword_result['listed_in'])
                result = result[result['listed_in'].str.contains(l, na=False, case=False)]
            if keyword_result['release_year']:
                result = result[result['release_year'] == keyword_result['release_year']]
            if keyword_result['year_from'] and keyword_result['year_to']:
                result = result[result['release_year'].between(keyword_result['year_from'], keyword_result['year_to'])]

            # 바뀐 부분: str.contains로 걸러내던 걸 임베딩 점수로 정렬하는 방식으로 교체
            if keyword_result['description']:
                result = self.rank_by_description(result, keyword_result['description'], user_input)
            else:
                result = result.head(TOP_K)

            try:
                data = result.to_dict(orient='records')

                with open("/Users/d.h/dev/git-archive/personal/gemini_chatbot/extract_from_netflex.json", mode='w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

            except Exception:
                logger.exception('error')

            else:
                print('Done!')
                return data

        data = extract_input(keyword_result)

        def answer(data, user_input):
            prompt = f"""
            사용자 질문: {user_input}
            {data}안에 있는 데이터만으로 답변을 할거야.
            score가 높은 순서대로 정렬되어 있어.
            그냥 보기좋고 간결하게 정리해서 보여줘"""

            try:
                response = client.models.generate_content(
                    model='gemini-3.5-flash-lite',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        ),
                        )
                print(response.text)
            except Exception:
                logger.exception("error: ")

        answer(data, user_input)


if __name__ == '__main__':
    a = Ai()
    while True:
        user_input = input("user👤: ")
        if user_input.lower() in ("q", "quit", "exit"):
            print("Exiting the program.")
            break
        a.extract_keyword(user_input)