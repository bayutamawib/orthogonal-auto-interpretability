import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 1. MAIN CONFIGURATION & HYPERPARAMETERS
# ==========================================
MODEL_NAME = "EleutherAI/pythia-160m"
SAMPLE_SIZE = 500 # Limited samples per dataset for fast Proof of Concept

# LoRA Configuration
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LEARNING_RATE = 1e-5
EPOCHS = 3
MAX_SEQ_LENGTH = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ==========================================
# 2. DATASET LOADING & PROCESSING
# ==========================================
def prepare_datasets():
    print("\n📦 Loading and processing datasets...")
    training_texts = []

    # Automatic execution environment detection (Kaggle vs. Local)
    if os.path.exists("/kaggle/input"):
        # ATTENTION: Change 'ortho-selfie-raw-data' to your Kaggle dataset folder name if different
        RAW_DATA_DIR = "/kaggle/input/datasets/narendrabayutama/ortho-groupsae-selfie-poc/data"
        OUTPUT_DIR = "/kaggle/input/datasets/narendrabayutama/ortho-groupsae-selfie-poc/models/lora_adapter"
        print(f"🌍 Running in Kaggle mode. Using data path: {RAW_DATA_DIR}")
    else:
        # Strict alignment with our established repo_structure
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
        OUTPUT_DIR = os.path.join(BASE_DIR, "models", "lora_adapter")
        print(f"💻 Running in Local mode. Using data path: {RAW_DATA_DIR}")

    # A. Empathy Dataset (ESConv)
    esconv_path = os.path.join(RAW_DATA_DIR, "ESConv.json")
    try:
        if os.path.exists(esconv_path):
            with open(esconv_path, "r", encoding="utf-8") as f:
                esconv_data = json.load(f)
                count = 0
                for item in esconv_data:
                    if count >= SAMPLE_SIZE: break
                    if "dialog" in item:
                        for turn in item["dialog"]:
                            if turn.get("speaker") == "supporter":
                                text = turn.get("content", "").strip()
                                if text:
                                    training_texts.append(f"Therapist: {text}")
                                    count += 1
                                    if count >= SAMPLE_SIZE: break
            print(f"✅ ESConv (Empathy): {count} samples loaded.")
        else:
            print(f"⚠️ File ESConv.json not found at {esconv_path}, skipping.")
    except Exception as e:
        print(f"❌ Failed to load ESConv: {e}")

    # B. Clinical Dataset (Heliosbrahma)
    helios_path = os.path.join(RAW_DATA_DIR, "heliosbrahma_dataset.json")
    try:
        if os.path.exists(helios_path):
            with open(helios_path, "r", encoding="utf-8") as f:
                count = 0
                for line in f:
                    if count >= SAMPLE_SIZE: break
                    line = line.strip()
                    if not line: continue
                    data = json.loads(line)
                    text_block = data.get("text", "")
                    if "<ASSISTANT>:" in text_block:
                        ast = text_block.split("<ASSISTANT>:")[1].strip()
                        training_texts.append(f"Clinical Diagnosis: {ast}")
                        count += 1
            print(f"✅ Heliosbrahma (Clinical): {count} samples loaded.")
        else:
             print(f"⚠️ File heliosbrahma_dataset.json not found at {helios_path}, skipping.")
    except Exception as e:
         print(f"❌ Failed to load Heliosbrahma: {e}")

    # C. Clinical Classification Dataset (Hugging Face)
    try:
        print("Downloading Clinical Classification dataset from Hugging Face...")
        hf_dataset = load_dataset("sai1908/Mental_Health_Condition_Classification", split="train")
        hf_texts = hf_dataset['text'][:SAMPLE_SIZE]
        hf_labels = hf_dataset['status'][:SAMPLE_SIZE]

        for i in range(len(hf_texts)):
            formatted_text = f"Patient Complaint: {hf_texts[i]}\nAnalysis: The patient exhibits symptoms of {hf_labels[i]}."
            training_texts.append(formatted_text)

        print(f"✅ Mental Health Classification (Clinical): {len(hf_texts)} samples loaded.")
    except Exception as e:
        print(f"❌ Failed to load HF classification dataset: {e}")

    return Dataset.from_dict({"text": training_texts}), OUTPUT_DIR

# ==========================================
# 3. MODEL & TOKENIZER INITIALIZATION
# ==========================================
print(f"\n🧠 Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" 
)

# ==========================================
# 4. LoRA CONFIGURATION
# ==========================================
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["query_key_value"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ==========================================
# 5. TRAINING (FINE-TUNING)
# ==========================================
train_dataset, output_dir = prepare_datasets()
print(f"\nTotal combined training data: {len(train_dataset)} rows.")

# Tokenize dataset explicitly (replaces SFTTrainer's automatic tokenization)
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("🔤 Tokenizing dataset...")
tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
    fp16=True if torch.cuda.is_available() else False, 
    report_to="none",
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("\n🚀 Starting LoRA Fine-Tuning process...")
trainer.train()

# ==========================================
# 6. SAVING TRAINED ADAPTER
# ==========================================
print("\n💾 Saving adapter model...")
os.makedirs(output_dir, exist_ok=True)
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Phase 1 Completed! LoRA model saved in directory: {os.path.abspath(output_dir)}")