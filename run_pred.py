import json

from langchain.prompts import PromptTemplate

from llm_ift_trials.llm import OpenAI
from llm_ift_trials.template import ADDR_OPENAI


def is_correct(pred, expected):
    pred_dict = json.loads(pred)

    return pred_dict == expected


def main():
    llm = OpenAI()

    address = "宜蘭縣三星鄉人和一路"
    expected = {
        "city": "宜蘭縣",
        "town": "三星鄉",
        "road": "人和一路",
    }

    prompt = PromptTemplate.from_template(ADDR_OPENAI).format(address=address)
    pred = llm.complete(prompt)

    print(f"Input Address: {address}")
    print(f"Output Prediction: {pred}")
    print(is_correct(pred, expected))


if __name__ == "__main__":
    main()
