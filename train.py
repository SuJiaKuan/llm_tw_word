import argparse

import torch
from langchain.prompts import PromptTemplate
from datasets import Dataset
from transformers import LlamaTokenizerFast
from transformers import LlamaForCausalLM
from transformers import TrainingArguments
from transformers.utils import logging
from trl import SFTTrainer

from llm_tw_word.translate import USER_PROMPT_TEMPLATE
from llm_tw_word.translate import ASSISTANT_PROMPT_TEMPLATE
from llm_tw_word.translate import SYSTEM_PROMPT
from llm_tw_word.io import load_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for model training (instruction finetuning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset/train.json",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name for training",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.2,
        help="Ratio of validation set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=55688,
        help="Random seed",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output_train",
        help="Path to output directory",
    )

    args = parser.parse_args()

    return args


def format_text(tokenizer, text_trad, text_tw, tokenize=False):
    user_text = PromptTemplate.from_template(
        USER_PROMPT_TEMPLATE,
    ).format(text_trad=text_trad)
    assistant_text = PromptTemplate.from_template(
        ASSISTANT_PROMPT_TEMPLATE,
    ).format(text_tw=text_tw)

    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT,
    }, {
        "role": "user",
        "content": user_text,
    }, {
        "role": "assistant",
        "content": assistant_text,
    }]
    formatted_data = tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=False,
    )

    return formatted_data


def create_dataset(tokenizer, samples):
    texts = []
    for sample in samples:
        text_trad = sample["text_trad"]
        text_tw = sample["text_tw"]
        texts.append(format_text(tokenizer, text_trad, text_tw))

    return Dataset.from_dict({"text": texts})


def main(args):
    data_path = args.data
    model_name= args.model
    val_ratio = args.val
    seed = args.seed
    output_dir = args.output

    logging.set_verbosity_info()

    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)

    samples = load_json(data_path)
    dataset = create_dataset(tokenizer, samples)
    dataset = dataset.train_test_split(test_size=val_ratio, seed=seed)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=tokenizer.model_max_length,
        dataset_num_proc=2,
        packing=True,  # Packs short sequences together to save time!
        args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            num_train_epochs=1,
            learning_rate=2e-5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            # optim="adamw_8bit",
            weight_decay=0.1,
            lr_scheduler_type="linear",
            seed=seed,
            output_dir=output_dir,
        ),
    )

    trainer.train()


if __name__ == "__main__":
    main(parse_args())
