import os

from langchain_openai import ChatOpenAI

from llm_tw_word.compute import memory


class OpenAI(object):

    def __init__(self, cached=True):
        if cached:
            self.complete = memory.cache(self.complete)

    def complete(self, prompt, model_name="gpt-3.5-turbo", temperature=0):
        model = ChatOpenAI(model_name=model_name, temperature=temperature)

        return model.invoke(prompt).content
