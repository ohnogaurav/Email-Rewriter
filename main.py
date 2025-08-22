# Step 1: Install necessary packages
!pip install transformers safetensors --quiet

# Step 2: Import libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Step 3: Set Hugging Face model repo
model_name = "ohnogaurav/email_rewriter_model"

# Step 4: Load tokenizer and model from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cpu", trust_remote_code=True)
model.eval()

# Step 5: Define test sentences
test_sentences = [
    "Hey, send me the report quick.",
    "Please complete the task by today.",
    "Can you check this document for errors?",
    "I need your input on the proposal."
]

# Step 6: Define tones
tones = ["Formal", "Polite", "Concise", "Funny"]

# Step 7: Generate rewrites
for sentence in test_sentences:
    print(f"\nOriginal: {sentence}")
    for tone in tones:
        # Add tone prefix if model was fine-tuned on tone-labeled inputs
        input_text = f"{tone}: {sentence}"
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate text with parameters to improve output length & creativity
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
