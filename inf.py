import argparse

from llm_tw_word.translate import TinyLlamaTranslator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for model inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "text",
        type=str,
        help="Text to be translated",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model name for inference",
    )

    args = parser.parse_args()

    return args


def main(args):
    translator = TinyLlamaTranslator(model=args.model)
    pred = translator.translate(args.text)

    print(f"Input: {args.text}")
    print(f"Output: {pred}")


if __name__ == "__main__":
    main(parse_args())
