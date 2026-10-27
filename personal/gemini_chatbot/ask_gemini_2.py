from log_config import logger
from google import genai
from google.genai import types
import os
import re
import pandas as pd
import json
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()


class SearchQuery(BaseModel):
    type: str | None = Field(
        description='"Movie" 또는 "TV Show". 언급이 없으면 빈 리스트'
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
        description=f"장르 목록. 언급이 없으면 빈 리스트"
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
        self.api_key = os.environ["GEMINI_API_KEY"]

        self.df = pd.read_csv(self.netflex_csv_path, index_col=0)
        self.df = self.df.fillna('Unknown')

        self.genres = sorted(
            self.df['listed_in'].str.split(',').explode().str.strip().unique()
            )

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)

    def extract_keyword(self, user_input):
        try:
            client = genai.Client(api_key=self.api_key)
        except Exception:
            logger.exception('Wrong api key')
            return

        prompt = (
            f"""
            너는 {user_input}으로 받은 단어를 가지고 분류를 할거야.
            기본적으로 자연어스러운 입력으로 부터 키워드를 분류해서 알맞은 넷플릭스 프로그램을 찾아주는일을 하는거야.
            데이터는 모두 영어로 되어있으니 한글로 받아도 영어로 바꿔줘.
            {self.genres}안에 포함된것으로만 장르로 구별하고 포함이 안되면 설명이나 프로그램 이름에 포함될수도 있어."""
        )

        try:
            response = client.models.generate_content(
                #model='gemini-3.8-flash',
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
            if keyword_result['description']:
                d = "|".join(keyword_result['description'])
                result = result[result['description'].str.contains(d, na=False, case=False)]

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
            {data}안에 있는 데이터만으로 답변을 할거야.
            그냥 보기좋고 간결하게 정리해서 보여줘"""

            try:
                response = client.models.generate_content(
                    #model='gemini-3.8-flash',
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


class test:
    def poo(self):
        df = pd.read_csv(self.netflex_csv_path, index_col=0)
        df = df.fillna('Unknown')

        war = df[df['description'].str.contains(r'\banimal\b', case=False)]

        words = (
            war['description'].str.lower()
            .str.findall(r'[a-z]+')      # 단어만 뽑기
            .explode()
            .value_counts()
        )

        stop = {'the', 'a', 'an', 'and', 'of', 'to', 'in', 'his', 'her', 'with', 'war', 'as', 'for', 'on', 'a', 'unknown'}
        print(words[~words.index.isin(stop)].head(30))

if __name__ == '__main__':
    a = Ai()
    while True:
        user_input = input("user👤: ")
        if user_input.lower() in ("q", "quit", "exit"):
            print("Exiting the program.")
            break
        a.extract_keyword(user_input)
    #t = test()
    #t.poo()