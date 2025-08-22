# Email Rewriter Model

This repository contains a text rewriting model that can rewrite sentences in multiple tones/styles, such as Formal, Polite, Concise, and Funny. The model is based on a fine-tuned Flan-T5 architecture and is hosted on Hugging Face.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Download the Model](#download-the-model)
- [Running Inference](#running-inference)
- [Training (Optional)](#training-optional)
- [Folder Structure](#folder-structure)
- [License](#license)

## Features
- Rewrites input sentences into different tones/styles.
- Easy to use with just Python and Transformers.
- Can be run locally or in Colab.

## Requirements
- Python 3.8+
- PyTorch 2.0+
- Hugging Face Transformers
- Safetensors
- Hugging Face Hub

## Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/ohnogaurav/email_rewriter_project.git
cd email_rewriter_project
pip install -r requirements.txt
```

## Download the Model
The trained model is hosted on Hugging Face. You can download it automatically by running the code, or manually if needed.

- **Hugging Face Hub:** [https://huggingface.co/ohnogaurav/email_rewriter_model](https://huggingface.co/ohnogaurav/email_rewriter_model)

> No Google Drive access is required. The model will be cached locally automatically when running for the first time.

## Running Inference
Use `main.py` to test the model with your own sentences. Example usage:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model from Hugging Face
model_name = "ohnogaurav/email_rewriter_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()

# Define test sentences and tones
sentences = [
    "Hey, send me the report quick.",
    "Please complete the task by today."
]
tones = ["Formal", "Polite", "Concise", "Funny"]

# Generate rewrites
for sentence in sentences:
    print(f"\nOriginal: {sentence}")
    for tone in tones:
        input_text = f"{tone}: {sentence}"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=60,
            num_beams=5,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        rewritten_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{tone}: {rewritten_text}")
```

### Notes
- Internet is required the first time you run the code to download the model from Hugging Face. After that, it works offline.
- The code works on CPU or GPU. GPU is optional but faster.

## Training (Optional)
You can fine-tune or train your own version using `train.py`. Make sure your training dataset follows the CSV format:

| input | style | output |
|-------|-------|--------|
| Your text here | Formal/Polite/Concise/Funny | Rewritten text |

Run training:

```bash
python train.py
```

Training will save the model to a specified folder, which can then be used for inference.

## Folder Structure
```
email_rewriter_project/
├── train.py           # Training script
├── main.py            # Inference/testing script
├── data/              # Sample datasets (optional)
│   └── data.csv
├── README.md          # Project instructions
├── requirements.txt   # Python dependencies
└── .gitignore         # Ignore unnecessary files
```

## License
This project is open-source. See LICENSE file for details.

