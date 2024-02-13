import argparse
import os

from datasets import load_dataset

from llm_tw_word.data import simp2data
from llm_tw_word.io import mkdir_p
from llm_tw_word.io import save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for dataset generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=2000,
        help="Number of training data",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=1000,
        help="Number of testing data",
    )
    parser.add_argument(
        "--altered_ratio",
        type=float,
        default=0.5,
        help="Ratio of altered data (i.e, the traditional text is different "
             "from taiwanese text)",
    )
    parser.add_argument(
        "--per_article_sentences",
        type=int,
        default=2,
        help="Number of max sentences to be extracted from an article",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=55688,
        help="Number of testing data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="generated_dataset",
        help="Path to output directory",
    )

    args = parser.parse_args()

    return args


def extract_sentence(
    article,
    min_len=30,
    sep_main="\n",
    sep_sub="。",
):
    paragraphs = article.split(sep_main)

    sentences = []
    for paragraph in paragraphs:
        sentences_ = paragraph.split(sep_sub)
        for idx, sentence in enumerate(sentences_):
            if idx != (len(sentences_) - 1):
                sentence = f"{sentence}{sep_sub}"
            sentences.append(sentence.strip())

    sentences = list(filter(
        lambda sent: len(sent) >= min_len,
        sentences,
    ))

    sentences = sorted(sentences, key=len, reverse=True)

    return sentences


def samples2dataset(samples, output_path):
    data = [
        {"text_trad": text_trad, "text_tw": text_tw}
        for text_trad, text_tw in samples
    ]
    save_json(output_path, data)


def main(args):
    mkdir_p(args.output)

    num_train_altered = int(args.num_train * args.altered_ratio)
    num_train_kept = args.num_train - num_train_altered
    num_test_altered = int(args.num_test * args.altered_ratio)
    num_test_kept = args.num_test - num_test_altered
    num_altered = num_train_altered + num_test_altered
    num_kept = num_train_kept + num_test_kept

    samples_altered = []
    samples_kept = []

    raw_ds = load_dataset("MBZUAI/Bactrian-X", "zh")["train"]

    for _ in range(5):
        raw_ds = raw_ds.shuffle(args.seed)

    for row in raw_ds:
        article = row["output"]
        sentences = extract_sentence(article)

        for text_simp in sentences[:args.per_article_sentences]:
            text_trad, text_tw = simp2data(text_simp)
            sample = (text_trad, text_tw)
            is_altered = text_trad != text_tw

            if is_altered:
                if len(samples_altered) < num_altered:
                    samples_altered.append(sample)
            else:
                if len(samples_kept) < num_kept:
                    samples_kept.append(sample)

            print(
                f"Generated Samples: {len(samples_altered) + len(samples_kept)}"
                f" / {num_altered + num_kept}",
            )

        if (
            (len(samples_altered) == num_altered)
            and (len(samples_kept) == num_kept)
        ):
            break

    samples_train = \
        samples_altered[:num_train_altered] + samples_kept[:num_train_kept]
    samples_test = \
        samples_altered[num_train_altered:] + samples_kept[num_train_kept:]

    samples2dataset(samples_train, os.path.join(args.output, "train.json"))
    samples2dataset(samples_test, os.path.join(args.output, "test.json"))

    print(f"Generated datasets saved in {args.output}")


if __name__ == "__main__":
    main(parse_args())
