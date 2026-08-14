import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# ==========================================
# 1. CONFIGURATION AND CLASSES
# ==========================================
# 🚀 Kaggle Optimization: Automatic GPU (CUDA) detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

MODEL_NAME = "EleutherAI/pythia-160m"

# 📁 Kaggle Path: Adjusted based on your previous execution logs
# which saved the output to the /kaggle/working/models/group_sae/ folder
SAE_PATH = "/kaggle/input/datasets/narendrabayutama/ortho-groupsae-selfie-poc/models/group_sae/group_sae_clinical_group_2.pt"

# (Note: If your file is located in the dataset folder, please change the path above 
# to something like: "/kaggle/input/your-dataset-name/group_sae_clinical_group_2.pt")

class TopK_SAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.k = k
        self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=True)

# Trigger texts
empathy_texts = ["I understand how overwhelming this feels.", "Your feelings are valid.", "I am here for you."]
clinical_texts = ["Patient shows signs of major depressive disorder.", "Symptoms of generalized anxiety.", "Diagnosis of OCD."]

def get_vectors(text, llm, tokenizer, sae):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llm(**inputs, output_hidden_states=True)
        # 1. Raw Hidden State dari Pythia
        raw_hidden = outputs.hidden_states[-1][0, -1, :] 
        raw_hidden = raw_hidden.to(torch.float32)
        
        # 2. Latent Feature dari SAE Encoder (HANYA LINEAR PROJECTION)
        activations = sae.encoder(raw_hidden)
        
        # 3. FIX: Terapkan Top-K Sparsity (Mematikan 99% neuron)
        k = sae.k
        topk_vals, topk_indices = torch.topk(activations, k, dim=-1)
        sae_latent = torch.zeros_like(activations)
        sae_latent.scatter_(-1, topk_indices, topk_vals)
        
        # 4. Terapkan ReLU (Memastikan tidak ada nilai negatif yang bocor)
        sae_latent = F.relu(sae_latent)
        
        return raw_hidden, sae_latent

def main():
    print("🧠 Loading Model and Tokenizer to GPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    llm.eval()

    print(f"📦 Loading SAE file from: {SAE_PATH}")
    sae = TopK_SAE(768, 768 * 16, 128).to(device)
    
    # Kaggle-specific Error Handling to prevent crashes due to path typos
    try:
        sae.load_state_dict(torch.load(SAE_PATH, map_location=device))
        print("✅ SAE file loaded successfully!")
    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found at {SAE_PATH}")
        print("💡 Solution: Check the 'Data' panel on the right side of your Kaggle workspace.")
        print("If you have restarted the session, files in /kaggle/working/ might be lost.")
        print("Point the path to /kaggle/input/... (your dataset location).")
        return
        
    sae.eval()

    # Extraction
    print("🔍 Running Vector Extraction...")
    raw_emp_list, sae_emp_list = [], []
    for t in empathy_texts:
        raw, latent = get_vectors(t, llm, tokenizer, sae)
        raw_emp_list.append(raw)
        sae_emp_list.append(latent)
        
    raw_clin_list, sae_clin_list = [], []
    for t in clinical_texts:
        raw, latent = get_vectors(t, llm, tokenizer, sae)
        raw_clin_list.append(raw)
        sae_clin_list.append(latent)

    # Average the vectors to get 1 Main Concept Vector
    mean_raw_emp = torch.stack(raw_emp_list).mean(dim=0)
    mean_raw_clin = torch.stack(raw_clin_list).mean(dim=0)
    
    mean_sae_emp = torch.stack(sae_emp_list).mean(dim=0)
    mean_sae_clin = torch.stack(sae_clin_list).mean(dim=0)

    # Calculate Cosine Similarity: S_C = (A . B) / (||A|| ||B||)
    sim_before = F.cosine_similarity(mean_raw_emp, mean_raw_clin, dim=0).item()
    sim_after = F.cosine_similarity(mean_sae_emp, mean_sae_clin, dim=0).item()

    print("\n" + "="*50)
    print("📊 ORTHOGONALIZATION ABLATION TEST RESULTS")
    print("="*50)
    print(f"Cosine Similarity BEFORE (Raw Pythia) : {sim_before:.4f}")
    print(f"Cosine Similarity AFTER (Group-SAE)  : {sim_after:.4f}")
    print("="*50)
    print("💡 Target: If the AFTER value is close to 0.0, it means your penalty SUCCESSFULLY split the dimensions to be orthogonal (90 degrees)!")

if __name__ == "__main__":
    main()