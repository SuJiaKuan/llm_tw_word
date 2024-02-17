# (WIP) LLM Taiwanese Words Translator

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
