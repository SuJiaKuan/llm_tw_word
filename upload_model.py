import argparse

from transformers import LlamaForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for uploading model to Hugging Face",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to model directory",
    )
    parser.add_argument(
        "target",
        type=str,
        help="Uploading target on Hugging Face",
    )

    args = parser.parse_args()

    return args


def main(args):
    model = LlamaForCausalLM.from_pretrained(args.model)
    model.push_to_hub(args.target)


if __name__ == "__main__":
    main(parse_args())
