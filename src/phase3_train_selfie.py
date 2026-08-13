import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. MAIN CONFIGURATION & HYPERPARAMETERS
# ==========================================
MODEL_NAME = "EleutherAI/pythia-160m"

# Smart Path Resolver for Kaggle/Local Execution
if os.path.exists("./models/lora_adapter/adapter_config.json"):
    LORA_DIR = "../models/lora_adapter"
    SAE_DIR = "../models/group_sae"
    OUTPUT_DIR = "../models/selfie_adapter"
    print(f"🌍 Running in Kaggle (Same Session).")
elif os.path.exists("/kaggle/input"):
    # ATTENTION: Adjust these paths to your actual Kaggle dataset names if running in a new session
    LORA_DIR = "/kaggle/input/datasets/narendrabayutama/ortho-groupsae-selfie-poc/models/lora_adapter"
    SAE_DIR = "/kaggle/input/datasets/narendrabayutama/ortho-groupsae-selfie-poc/models/group_sae"
    OUTPUT_DIR = "/kaggle/input/datasets/narendrabayutama/ortho-groupsae-selfie-poc/models/selfie_adapter"
    print(f"🌍 Running in Kaggle (Attached Datasets).")
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LORA_DIR = os.path.join(BASE_DIR, "models", "lora_adapter")
    SAE_DIR = os.path.join(BASE_DIR, "models", "group_sae")
    OUTPUT_DIR = os.path.join(BASE_DIR, "models", "selfie_adapter")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters based on Group-SAE literature
EXPANSION_FACTOR = 16 
TOP_K = 128
INPUT_DIM = 768  
HIDDEN_DIM = INPUT_DIM * EXPANSION_FACTOR

# SelfIE Adapter Hyperparameters
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 8
NUM_TRAIN_SAMPLES = 500 # Using a subset of latents for fast PoC execution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ==========================================
# 2. ARCHITECTURES
# ==========================================
class TopK_SAE(nn.Module):
    """Sparse Autoencoder architecture to load Phase 2 weights."""
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

class ScalarAffineAdapter(nn.Module):
    """
    Lightweight adapter to transform SAE latent vectors into LLM activation space.
    Contains only d_model + 1 parameters (Scale and Bias).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, h):
        """ Transforms the latent vector h -> f(h) """
        return (self.scale * h) + self.bias

# ==========================================
# 3. LATENT EXTRACTION & AUTO-INTERPRETABILITY
# ==========================================
def extract_sae_vectors(sae_path, device):
    """Extracts the actual geometric feature directions from the SAE decoder."""
    print(f"\n🔍 Extracting real latent vectors from Group-SAE...")
    sae = TopK_SAE(INPUT_DIM, HIDDEN_DIM, TOP_K)
    
    try:
        sae.load_state_dict(torch.load(sae_path, map_location=device))
    except Exception as e:
        print(f"❌ Failed to load SAE weights at {sae_path}. Error: {e}")
        exit()
        
    # The decoder weights map hidden_dim -> input_dim. 
    # Shape of weight is (input_dim, hidden_dim). Transposing gives us (hidden_dim, input_dim),
    # where each row is a 768-dimensional latent vector (h).
    latent_vectors = sae.decoder.weight.detach().T
    print(f"✅ Extracted {latent_vectors.shape[0]} latent vectors of dimension {latent_vectors.shape[1]}.")
    return latent_vectors

def generate_labels_with_explainer(num_samples):
    """
    Simulates the Auto-Interpretability pipeline.
    In production, this queries Groq API with activating texts to get real labels.
    """
    print("\n📝 Running Auto-Interpretability Pipeline (Explainer LLM)...")
    text_labels = []
    api_key = os.environ.get("GROQ_API_KEY")
    
    for i in range(num_samples):
        if api_key:
            # Future Implementation: Call Groq API here using llama-3.3-70b-versatile
            pass
        
        # Fallback for PoC to ensure Cross-Entropy training runs smoothly
        desc = f"Concept representation for clinical or empathetic feature {i}"
        
        # Append closing quote and EOS token (Crucial for SelfIE training format)
        formatted_label = f"{desc}\"<|endoftext|>" 
        text_labels.append(formatted_label)
        
    print(f"✅ Generated {num_samples} explanation labels.")
    return text_labels

# ==========================================
# 4. TRAINING LOOP (FIXED COMPUTATION GRAPH)
# ==========================================
def train_selfie_adapter(model, tokenizer, adapter, latent_vectors, text_labels, device):
    print("\n🚀 Starting Trained SelfIE Adapter Training...")
    
    # Ensure Base LLM is completely frozen to preserve general capabilities
    for param in model.parameters():
        param.requires_grad = False
    model.eval() 
    
    adapter.train()
    optimizer = optim.AdamW(adapter.parameters(), lr=LEARNING_RATE)
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    # SelfIE Prompt Template
    prompt_template = "The following latent feature represents: \""
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        
        for i in range(0, NUM_TRAIN_SAMPLES, BATCH_SIZE):
            # Slicing the real extracted SAE vectors
            batch_h = latent_vectors[i:i+BATCH_SIZE].to(device)
            batch_labels = text_labels[i:i+BATCH_SIZE]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            # ⚠️ CRITICAL FIX: Ambil ukuran terkecil agar sisa batch terakhir tidak error
            current_batch_size = min(len(batch_h), len(batch_labels))
            
            for j in range(current_batch_size):
                h = batch_h[j]
                target_text = batch_labels[j]
                
                # Tokenize prompt and target label
                prompt_ids = tokenizer.encode(prompt_template, return_tensors="pt").to(device)
                target_ids = tokenizer.encode(target_text, return_tensors="pt").to(device)
                full_input_ids = torch.cat([prompt_ids, target_ids], dim=1)
                
                # Get standard embeddings (No grad needed just to fetch the base embeddings)
                with torch.no_grad():
                    base_embeds = model.get_input_embeddings()(full_input_ids)
                
                # Clone the embeddings to maintain a safe computation graph
                inputs_embeds = base_embeds.clone()
                
                # Inject transformed activation f(h) at the placeholder position (' "')
                placeholder_idx = prompt_ids.shape[1] - 1
                f_h = adapter(h)
                inputs_embeds[0, placeholder_idx, :] = f_h
                
                # Forward pass WITHOUT torch.no_grad()
                outputs = model(inputs_embeds=inputs_embeds)
                
                logits = outputs.logits[0]
                
                # Calculate Cross-Entropy Loss on target tokens
                shift_logits = logits[placeholder_idx:-1, :].contiguous()
                shift_labels = target_ids[0].contiguous()
                
                loss = cross_entropy_loss(shift_logits, shift_labels)
                batch_loss += loss
            
            # Jangan lupa bagi loss dengan current_batch_size yang baru
            batch_loss = batch_loss / current_batch_size
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            
        avg_loss = total_loss / (NUM_TRAIN_SAMPLES / BATCH_SIZE)
        print(f"  Epoch {epoch+1}/{EPOCHS} | Average Cross-Entropy Loss: {avg_loss:.4f}")

    print("\n✅ Training Complete!")
    return adapter

# ==========================================
# 5. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print(f"\n🧠 Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"🧠 Loading Base Model ({MODEL_NAME})...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    print(f"🔗 Attaching LoRA Adapter from Phase 1 ({LORA_DIR})...")
    try:
        fine_tuned_model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
        print("✅ Fine-tuned LLM loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load LoRA Adapter. Error: {e}")
        exit()
        
    # Initialize the lightweight Scalar Affine Adapter
    adapter = ScalarAffineAdapter(hidden_dim=INPUT_DIM).to(device)
    
    # Extract real vectors from Phase 2 Group-SAE (Using Group 2 as an example)
    target_sae_file = os.path.join(SAE_DIR, "group_sae_clinical_group_2.pt")
    latent_vecs = extract_sae_vectors(target_sae_file, device)
    
    # Generate labels using the Explainer pipeline
    text_lbls = generate_labels_with_explainer(NUM_TRAIN_SAMPLES)
    
    # Train the Adapter
    trained_adapter = train_selfie_adapter(
        model=fine_tuned_model,
        tokenizer=tokenizer,
        adapter=adapter,
        latent_vectors=latent_vecs,
        text_labels=text_lbls,
        device=device
    )
    
    # Save the Adapter weights
    save_path = os.path.join(OUTPUT_DIR, "trained_selfie_adapter.pt")
    torch.save(trained_adapter.state_dict(), save_path)
    print(f"💾 Trained SelfIE Adapter successfully saved to {save_path}")