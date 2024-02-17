import argparse

from llm_tw_word.const import TRANSLATOR_TYPE
from llm_tw_word.const import DEFAULT_LLAMA_MODEL
from llm_tw_word.const import DEFAULT_OPENAI_MODEL
from llm_tw_word.translate import LlamaTranslator
from llm_tw_word.translate import OpenAITranslator


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

    args = parser.parse_args()

    return args


def main(args):
    text_trad = args.text
    translator_name = args.translator
    model_name = args.model

    if translator_name == TRANSLATOR_TYPE.LLAMA:
        model_name = model_name if model_name else DEFAULT_LLAMA_MODEL
        translator = LlamaTranslator(model_name=model_name)
    else:
        model_name = model_name if model_name else DEFAULT_OPENAI_MODEL
        translator = OpenAITranslator(model_name=model_name)

    pred = translator.translate([text_trad])[0]

    print(f"Translator: {translator_name}")
    print(f"Model: {model_name}")
    print(f"Input Text: {text_trad}")
    print(f"Output Text: {pred}")


if __name__ == "__main__":
    main(parse_args())
