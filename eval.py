import argparse
import os

import editdistance
import numpy as np
from tqdm import tqdm

from llm_tw_word.translate import OpenAITranslator
from llm_tw_word.translate import LlamaTranslator
from llm_tw_word.const import TRANSLATOR_TYPE
from llm_tw_word.const import DEFAULT_LLAMA_MODEL
from llm_tw_word.const import DEFAULT_OPENAI_MODEL
from llm_tw_word.io import load_json
from llm_tw_word.io import save_json
from llm_tw_word.io import mkdir_p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for model performance evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "translator",
        type=str,
        choices=(
            TRANSLATOR_TYPE.LLAMA,
            TRANSLATOR_TYPE.OPENAI,
        ),
        help="Translator type",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specified model name for the translator. If not provided, there"
             " will be a default model",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset/test.json",
        help="Path to dataset file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output_eval",
        help="Path to output directory",
    )

    args = parser.parse_args()

    return args


def main(args):
    data_path = args.data
    translator_name = args.translator
    model_name = args.model
    output_dir = args.output

    mkdir_p(output_dir)

    if translator_name == TRANSLATOR_TYPE.LLAMA:
        model_name = model_name if model_name else DEFAULT_LLAMA_MODEL
        translator = LlamaTranslator(model_name=model_name)
    else:
        model_name = model_name if model_name else DEFAULT_OPENAI_MODEL
        translator = OpenAITranslator(model_name=model_name)

    samples = load_json(data_path)

    print(f"Running translator: {translator_name}, model: {model_name}")

    results = []
    for sample in tqdm(samples):
        text_trad = sample["text_trad"]
        text_tw = sample["text_tw"]
        pred = translator.translate([text_trad])[0]
        distance = editdistance.eval(text_tw, pred)

        results.append({
            "text_trad": text_trad,
            "text_tw": text_tw,
            "pred": pred,
            "distance": distance,
        })

    avg_distance = np.mean([r["distance"] for r in results])

    output_path = os.path.join(output_dir, f"{translator_name}.json")
    save_json(output_path, results)

    print(f"Average of Edit Distance: {avg_distance}")
    print(f"Evaluation Results saved :{output_path}")


if __name__ == "__main__":
    main(parse_args())
