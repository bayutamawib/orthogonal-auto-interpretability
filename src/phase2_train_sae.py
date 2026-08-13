import os
import time
import json
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# ==========================================
# 1. MAIN CONFIGURATION & HYPERPARAMETERS
# ==========================================
MODEL_NAME = "EleutherAI/pythia-160m"

# Smart Path Resolver (Adapts to Kaggle session, Kaggle input, or Local execution)
if os.path.exists("../models/lora_adapter/adapter_config.json"):
    # Case A: Running in the same Kaggle session right after Phase 1
    RAW_DATA_DIR = "../data/raw" 
    LORA_DIR = "../models/lora_adapter"
    OUTPUT_DIR = "../models/group_sae"
    print(f"🌍 Running in Kaggle (Same Session). Output path: {OUTPUT_DIR}")
elif os.path.exists("/kaggle/input"):
    # Case B: Running in a new Kaggle session with datasets attached
    # ATTENTION: Adjust 'ortho-selfie-lora-pythia' to your actual dataset name if uploaded
    RAW_DATA_DIR = "/kaggle/input/datasets/narendrabayutama/ortho-groupsae-selfie-poc/data/raw" 
    LORA_DIR = "/kaggle/input/datasets/narendrabayutama/ortho-groupsae-selfie-poc/models/lora_adapter" 
    OUTPUT_DIR = "/kaggle/input/datasets/narendrabayutama/ortho-groupsae-selfie-poc/models/selfie_adapter"
    print(f"🌍 Running in Kaggle (Attached Dataset). Output path: {OUTPUT_DIR}")
else:
    # Case C: Strict alignment with local repo_structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    LORA_DIR = os.path.join(BASE_DIR, "models", "lora_adapter")
    OUTPUT_DIR = os.path.join(BASE_DIR, "models", "group_sae")
    print(f"💻 Running in Local mode. Output path: {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# SAE Hyperparameters (Aligned with Group-SAE literature)
EXPANSION_FACTOR = 16 
TOP_K = 128
INPUT_DIM = 768  # Pythia-160M hidden dimension
HIDDEN_DIM = INPUT_DIM * EXPANSION_FACTOR
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4

# Orthogonal Penalty (Separating Empathy vs. Clinical Reasoning)
ORTHO_LAMBDA = 10.0  

# Streaming Buffer Capacity (Safe for Kaggle RAM limitations)
BUFFER_TEXT_LIMIT = 500 
EPOCHS = 3
SAMPLE_SIZE = 500 # Limited texts per dataset for PoC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ==========================================
# 2. GROUP-SAE ARCHITECTURE
# ==========================================
class TopK_SAE(nn.Module):
    """Sparse Autoencoder using Top-K activation."""
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        topk_values, topk_indices = torch.topk(encoded, self.k, dim=-1)
        sparse_encoded = torch.zeros_like(encoded).scatter_(-1, topk_indices, topk_values)
        sparse_encoded = torch.relu(sparse_encoded)
        reconstructed = self.decoder(sparse_encoded)
        return reconstructed, sparse_encoded

def calculate_fvu(original, reconstructed):
    """Calculating Fraction of Variance Unexplained (FVU)."""
    mse = torch.nn.functional.mse_loss(reconstructed, original, reduction='mean')
    variance = torch.var(original, unbiased=False)
    fvu = mse / (variance + 1e-8) 
    return fvu.item()

# ==========================================
# 3. DATASET LOADING (EMPATHY VS CLINICAL)
# ==========================================
def load_datasets():
    print("\n=== Loading Datasets for Two Manifolds ===")
    
    empathy_texts = []
    clinical_texts = []

    # A. EMPATHY MANIFOLD (ESConv)
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
                                    empathy_texts.append(text)
                                    count += 1
                                    if count >= SAMPLE_SIZE: break
            print(f"✅ Empathy Manifold (ESConv): {len(empathy_texts)} samples.")
        else:
            print(f"⚠️ File ESConv.json not found, skipping.")
    except Exception as e:
        print(f"❌ Failed to load ESConv: {e}")

    # B. CLINICAL MANIFOLD (Heliosbrahma)
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
                        clinical_texts.append(ast)
                        count += 1
            print(f"✅ Clinical Manifold (Heliosbrahma): {count} samples.")
        else:
            print(f"⚠️ File heliosbrahma_dataset.json not found, skipping.")
    except Exception as e:
        print(f"❌ Failed to load Heliosbrahma: {e}")

    # C. ADDITIONAL CLINICAL MANIFOLD (Hugging Face)
    try:
        print("Downloading Clinical Classification dataset from Hugging Face...")
        hf_dataset = load_dataset("sai1908/Mental_Health_Condition_Classification", split="train")
        hf_texts = hf_dataset['text'][:SAMPLE_SIZE]
        clinical_texts.extend(hf_texts)
        print(f"✅ Clinical Manifold (HF Mental Health): {len(hf_texts)} samples.")
    except Exception as e:
        print(f"❌ Failed to load HF classification dataset: {e}")
        
    return empathy_texts, clinical_texts

# ==========================================
# 4. ACTIVATION EXTRACTION & TRAINING
# ==========================================
def get_token_activations(text, llm_model, tokenizer, device, group_layers):
    """Extract activations from specific layers dynamically."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = llm_model(**inputs, output_hidden_states=True)
        # Fetch hidden states, skip initial embedding layer (index 0 usually denotes embeddings)
        hidden_states = outputs.hidden_states[1:] 
        
        layer_acts = []
        for layer_idx in group_layers:
            state = hidden_states[layer_idx].squeeze(0).cpu()
            layer_acts.append(state)
            
        return torch.cat(layer_acts, dim=0)

def train_streaming_sae(actual_groups, empathy_texts, clinical_texts, llm_model, tokenizer, device):
    """Dynamic Buffer Training Cycle with Orthogonal Penalty."""
    
    for idx, group_layers in enumerate(actual_groups):
        group_id = idx + 1
        print(f"\n{'='*60}")
        print(f"=== Starting Streaming Group-SAE {group_id} ===")
        print(f"Target Layers: {group_layers}")
        
        sae_model = TopK_SAE(INPUT_DIM, HIDDEN_DIM, TOP_K).to(device)
        optimizer = optim.Adam(sae_model.parameters(), lr=LEARNING_RATE)
        mse_loss_fn = nn.MSELoss()
        sae_model.train()
        
        for epoch in range(EPOCHS):
            print(f"\n  --- Epoch {epoch+1}/{EPOCHS} ---")
            
            clin_iter = itertools.cycle(clinical_texts)
            emp_iter = itertools.cycle(empathy_texts)
            
            max_texts = max(len(clinical_texts), len(empathy_texts))
            text_processed = 0
            total_steps = 0  
            step_start_time = time.time() 
            
            buffer_clin = []
            buffer_emp = []
            
            while text_processed < max_texts:
                # 1. Activation Caching Phase (Filling the Buffer)
                while len(buffer_clin) < BUFFER_TEXT_LIMIT and text_processed < max_texts:
                    t_clin = next(clin_iter)
                    acts_c = get_token_activations(t_clin, llm_model, tokenizer, device, group_layers)
                    buffer_clin.append(acts_c)
                    
                    t_emp = next(emp_iter)
                    acts_e = get_token_activations(t_emp, llm_model, tokenizer, device, group_layers)
                    buffer_emp.append(acts_e)
                    
                    text_processed += 1
                    
                if not buffer_clin or not buffer_emp:
                    break
                    
                # 2. Training Phase (Draining the Buffer)
                print(f"  [Epoch {epoch+1}] Training from buffer... (Text Progress: {text_processed}/{max_texts})")
                tensor_clin = torch.cat(buffer_clin, dim=0)
                tensor_emp = torch.cat(buffer_emp, dim=0)
                
                # Shuffle tokens to prevent overfitting
                tensor_clin = tensor_clin[torch.randperm(tensor_clin.size(0))]
                tensor_emp = tensor_emp[torch.randperm(tensor_emp.size(0))]
                
                num_tokens = min(tensor_clin.size(0), tensor_emp.size(0))
                
                for i in range(0, num_tokens, BATCH_SIZE):
                    x_clin = tensor_clin[i:i+BATCH_SIZE].to(device, dtype=torch.float32)
                    x_emp = tensor_emp[i:i+BATCH_SIZE].to(device, dtype=torch.float32)
                    
                    if x_clin.size(0) < BATCH_SIZE:
                        continue
                        
                    optimizer.zero_grad()
                    
                    recon_clin, sparse_clin = sae_model(x_clin)
                    loss_clin = mse_loss_fn(recon_clin, x_clin)
                    
                    recon_emp, sparse_emp = sae_model(x_emp)
                    loss_emp = mse_loss_fn(recon_emp, x_emp)
                    
                    # Compute Cosine Similarity for Orthogonal Penalty
                    mean_clin = sparse_clin.mean(dim=0)
                    mean_emp = sparse_emp.mean(dim=0)
                    ortho_penalty = F.cosine_similarity(mean_clin.unsqueeze(0), mean_emp.unsqueeze(0), eps=1e-8).squeeze()
                    
                    # TOTAL LOSS = Clinical Recon + Empathy Recon + (Lambda * Overlap Penalty)
                    loss = loss_clin + loss_emp + (ORTHO_LAMBDA * ortho_penalty)
                    loss.backward()
                    optimizer.step()
                    
                    total_steps += 1
                    
                    if total_steps % 50 == 0:
                        elapsed_time = time.time() - step_start_time 
                        fvu_c = calculate_fvu(x_clin, recon_clin)
                        fvu_e = calculate_fvu(x_emp, recon_emp)
                        
                        print(f"    Step {total_steps:04d} | Loss: {loss.item():.4f} | Ortho Pen: {ortho_penalty.item():.4f} | FVU Clin: {fvu_c:.4f} | FVU Emp: {fvu_e:.4f} | Time/50 steps: {elapsed_time:.2f}s")
                        step_start_time = time.time() 
                        
                # 3. Clear RAM Buffer
                buffer_clin = []
                buffer_emp = []
        
        # Save SAE Weights
        save_path = os.path.join(OUTPUT_DIR, f"group_sae_clinical_group_{group_id}.pt")
        torch.save(sae_model.state_dict(), save_path)
        print(f"✅ Group-SAE {group_id} successfully saved to {save_path}")

# ==========================================
# 5. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print(f"\n🧠 Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"🧠 Loading Base Model ({MODEL_NAME})...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    
    print(f"🔗 Attaching LoRA Adapter from Phase 1 ({LORA_DIR})...")
    try:
        # Load the fine-tuned LoRA model
        llm_model = PeftModel.from_pretrained(base_model, LORA_DIR)
        llm_model.eval() # Ensure model is frozen for activation extraction
        print("✅ Fine-tuned LLM ready for extraction!")
    except Exception as e:
        print(f"❌ Failed to load LoRA Adapter. Did Phase 1 complete successfully? Error: {e}")
        exit()
    
    empathy_texts, clinical_texts = load_datasets()
    
    # Define Layer Groups Architecture
    # Pythia-160M has 12 layers (0 to 11). Grouping early vs middle-late layers.
    if empathy_texts and clinical_texts:
        actual_groups = [
            [0],                                       # Group 1: Layer 0
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]        # Group 2: Layers 1-11
        ]
        train_streaming_sae(actual_groups, empathy_texts, clinical_texts, llm_model, tokenizer, device)
    else:
        print("⚠️ Execution aborted: Datasets are empty or failed to load.")