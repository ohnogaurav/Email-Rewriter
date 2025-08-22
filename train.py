# ===== 1️⃣ Install & Import Dependencies =====
!pip install transformers datasets accelerate --quiet

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch

# ===== 2️⃣ Mount Google Drive =====
from google.colab import drive
drive.mount('/content/drive')

# ===== 3️⃣ Load Dataset =====
dataset_path = "/content/drive/MyDrive/RewriteGenAI/rewriter_train.csv"
dataset = load_dataset("csv", data_files={"train": dataset_path}, delimiter=",")

print("Dataset sample:", dataset["train"][0])

# ===== 4️⃣ Initialize Tokenizer & Model =====
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ===== 5️⃣ Preprocessing Function =====
def preprocess(batch):
    # Combine input + style as instruction
    inputs = [f"Rewrite in {s} style: {i}" for i, s in zip(batch["input"], batch["style"])]
    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")
    labels = tokenizer(batch["output"], max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset["train"].map(preprocess, batched=True)

# ===== 6️⃣ Training Arguments =====
training_args = TrainingArguments(
    output_dir="./email_rewriter_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available()  # use mixed precision if GPU available
)

# ===== 7️⃣ Initialize Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# ===== 8️⃣ Train Model =====
trainer.train()

# ===== 9️⃣ Save Model =====
trainer.save_model("/content/drive/MyDrive/RewriteGenAI/email_rewriter_model")
print("Model saved successfully!")

# ===== 🔟 Test / Inference Function =====
def rewrite_message(text, style):
    prompt = f"Rewrite in {style} style: {text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(**inputs, max_length=64, num_beams=5, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ===== 11️⃣ Example Test =====
test_msg = "Hey, send me the report."
print("Formal:", rewrite_message(test_msg, "formal"))
print("Polite:", rewrite_message(test_msg, "polite"))
print("Concise:", rewrite_message(test_msg, "concise"))
print("Funny:", rewrite_message(test_msg, "funny"))
