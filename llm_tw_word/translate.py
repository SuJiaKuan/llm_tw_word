import os

import torch
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import pipeline

from llm_tw_word.compute import memory


TEMPLATE_OPENAI = """\
Instruction: 對於輸入內容的中文文字，請將中國用語轉成台灣的用語，其他非中文文字或非中國用語都維持不變。

Input: ```這個視頻的質量真高啊```
Output: ```這個影片的品質真高啊```

Input: ```{text_trad}```
Output: \
"""

TEMPLATE_TINY_LLAMA = """\
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


class TinyLlamaTranslator(object):

    def __init__(self, model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def _complete(self, prompt, max_new_tokens=4096):
        messages = [{
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
        prompt = PromptTemplate.from_template(TEMPLATE_TINY_LLAMA).format(
            text_trad=text_trad,
        )
        pred = self._complete(prompt)

        if output_part in pred:
            pred = pred.split(output_part)[-1]

        pred = pred.replace(sep, "").strip().split("\n")[0]

        return pred
