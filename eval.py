import argparse
import os

import editdistance
import numpy as np
from tqdm import tqdm

from llm_tw_word.translate import OpenAITranslator
from llm_tw_word.translate import TinyLlamaTranslator
from llm_tw_word.io import load_json
from llm_tw_word.io import save_json
from llm_tw_word.io import mkdir_p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for model performance evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    mkdir_p(args.output)

    configs = [
        ("openai", OpenAITranslator()),
        ("tiny_llama", TinyLlamaTranslator()),
    ]

    samples = load_json(args.data)

    for model_name, translator in configs:
        print(f"Running Model: {model_name}")

        results = []
        for sample in tqdm(samples):
            text_trad = sample["text_trad"]
            text_tw = sample["text_tw"]
            pred = translator.translate(text_trad)
            distance = editdistance.eval(text_tw, pred)

            results.append({
                "text_trad": text_trad,
                "text_tw": text_tw,
                "pred": pred,
                "distance": distance,
            })

        save_json(os.path.join(args.output, f"{model_name}.json"), results)

        avg_distance = np.mean([r["distance"] for r in results])
        print(f"Average of Edit Distance: {avg_distance}")

    print(f"Evaluation Results saved in {args.output}")


if __name__ == "__main__":
    main(parse_args())
