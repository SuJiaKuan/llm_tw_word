# Taiwanese Words Translator by LLMs

The repository contains code and data for LLMs that translate China words to Taiwanese words. The main technique is instruction fine-tuning.

Example:
- Input: `這個軟件的質量真高啊`
- Output: `這個軟體的品質真高啊`

## Installation

* Install Miniconda or Anaconda

* Create a Conda environment: `tw_word`.
```bash
conda create --name tw_word python=3.10
```

* Activate the environment.
```bash
conda activate tw_word
```

* Install PyTorch related packages.
```bash
# GPU
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
# or, CPU-only (This may be very slow)
pip install torch==2.2.0 torchvision==0.17.0
```

* Install required packages.
```bash
pip install -r requirements.txt
```

* (Optional) Setup your OpenAI API key if you want to use OpenAI related functions.

```bash
export OPENAI_API_KEY=${YOUR_OPENAI_API_KEY}
```

## Text Translation

To run the translation powered by Llama translator, just typing following command on your terminal:
```bash
python inf.py "這個軟件的質量真高啊" llama --model "feabries/TaiwanWordTranslator-v0.1"
```

For OpenAI translator:
```bash
python inf.py "這個軟件的質量真高啊" openai
```

## Performance Evaluation

To run the testing set evaluation for llama translator:
```bash
python eval.py llama --model "feabries/TaiwanWordTranslator-v0.1"
```

For OpenAI translator:
```bash
python eval.py openai
```

## Model Training

To run llama model training on training set:
```bash
python train.py
```

## About the Dataset

Current dataset is collected from [MBZUAI/Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X) and automatically labeled by [繁化姬](https://zhconvert.org).
