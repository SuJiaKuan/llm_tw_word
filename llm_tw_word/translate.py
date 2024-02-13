import os

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from llm_tw_word.compute import memory


TEMPLATE_OPENAI = """\
Instruction: 對於輸入內容的中文文字，請將中國用語轉成台灣的用語，其他非中文文字或非中國用語都維持不變。

Input: ```這個視頻的質量真高啊```
Output: ```這個影片的品質真高啊```

Input: ```{text_trad}```
Output: \
"""


class OpenAITranslator(object):

    def __init__(self, cached=True):
        if cached:
            self._complete = memory.cache(self._complete)

    def _complete(self, prompt, model_name="gpt-3.5-turbo", temperature=0):
        model = ChatOpenAI(model_name=model_name, temperature=temperature)

        return model.invoke(prompt).content

    def translate(self, text_trad, sep="```"):
        prompt = PromptTemplate.from_template(TEMPLATE_OPENAI).format(
            text_trad=text_trad,
        )
        pred = self._complete(prompt).replace(sep, "")

        return pred
