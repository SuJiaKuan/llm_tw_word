import os

import torch
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from transformers import pipeline

from llm_tw_word.compute import memory


SYSTEM_PROMPT = """\
對於輸入內容的中文文字，請將中國用語轉成台灣的用語，其他非中文文字或非中國用語都維持不變。

範例：
Input: ```這個視頻的質量真高啊```
Output: ```這個影片的品質真高啊```\
"""

USER_PROMPT_TEMPLATE = """\
Input: ```{text_trad}```\
"""


class OpenAITranslator(object):

    def __init__(self, cached=True):
        if cached:
            self._complete = memory.cache(self._complete)

    def _complete(self, prompt, model_name="gpt-3.5-turbo", temperature=0):
        messages = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(prompt),
        ]
        model = ChatOpenAI(model_name=model_name, temperature=temperature)

        return model.invoke(messages).content

    def translate(self, text_trad, output_part="Output:", sep="```"):
        prompt = PromptTemplate.from_template(USER_PROMPT_TEMPLATE).format(
            text_trad=text_trad,
        )
        pred = self._complete(prompt)
        pred = pred.split(output_part)[-1].strip().replace(sep, "")

        return pred


class TinyLlamaTranslator(object):

    def __init__(self, model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def _complete(self, prompt, max_new_tokens=2048):
        messages = [{
            "role": "system",
            "content": SYSTEM_PROMPT,
        }, {
            "role": "user",
            "content": prompt,
        }]
        input_text = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = self.pipeline(
            input_text,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

        return outputs[0]["generated_text"][len(input_text):]

    def translate(self, text_trad, output_part="Output:", sep="```"):
        prompt = PromptTemplate.from_template(USER_PROMPT_TEMPLATE).format(
            text_trad=text_trad,
        )
        pred = self._complete(prompt)
        pred = pred.split(output_part)[-1].strip().replace(sep, "")

        return pred
