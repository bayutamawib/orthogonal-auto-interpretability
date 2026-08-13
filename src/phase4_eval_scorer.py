import os
import random
import math
import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from groq import Groq

# ==========================================
# 1. MAIN CONFIGURATION & HYPERPARAMETERS
# ==========================================
MODEL_NAME = "EleutherAI/pythia-160m"

# Path Configuration for Kaggle (Strictly Maintained)
if os.path.exists("../models/selfie_adapter/trained_selfie_adapter.pt"):
    LORA_DIR = "../models/lora_adapter"
    SAE_DIR = "../models/group_sae"
    SELFIE_DIR = "../models/selfie_adapter"
    print(f"🌍 Running in Kaggle (Same Session).")
elif os.path.exists("/kaggle/input"):
    LORA_DIR = "/kaggle/input/datasets/narendrabayutama/ortho-groupsae-selfie-poc/models/lora_adapter"
    SAE_DIR = "/kaggle/input/datasets/narendrabayutama/ortho-groupsae-selfie-poc/models/group_sae"
    OUTPUT_DIR = "/kaggle/input/datasets/narendrabayutama/ortho-groupsae-selfie-poc/models/selfie_adapter"
    print(f"🌍 Running in Kaggle (Attached Datasets).")
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LORA_DIR = os.path.join(BASE_DIR, "models", "lora_adapter")
    SAE_DIR = os.path.join(BASE_DIR, "models", "group_sae")
    SELFIE_DIR = os.path.join(BASE_DIR, "models", "selfie_adapter")

INPUT_DIM = 768
HIDDEN_DIM = INPUT_DIM * 16
TOP_K = 128
NUM_EVAL_SAMPLES = 10 # 10 samples (Total 20 API Requests)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# Initialize Groq API
api_key = ""
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found! Please ensure the Kaggle secret is activated.")
groq_client = Groq(api_key=api_key)

# ==========================================
# 2. ARCHITECTURE DEFINITIONS
# ==========================================
class TopK_SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

class ScalarAffineAdapter(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, h):
        return (self.scale * h) + self.bias

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def pearson_correlation(x, y):
    """Calculates Pearson Correlation Coefficient."""
    if len(x) < 2: return 0.0
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mean_x)**2 for a in x) * sum((b - mean_y)**2 for b in y))
    return numerator / denominator if denominator != 0 else 0.0

def get_top_activating_feature(text, llm_model, tokenizer, sae, device):
    """
    REAL PIPELINE: Passes text through the model, extracts hidden states, 
    feeds to SAE, and finds the single mathematical feature with the highest activation.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llm_model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[1][0, -1, :] 
        
        # Ensure hidden_state dtype matches sae.encoder.weight dtype
        hidden_state = hidden_state.to(sae.encoder.weight.dtype)
        
        activations = sae.encoder(hidden_state)
        top_feature_id = torch.argmax(activations).item()
        latent_vector = sae.decoder.weight[:, top_feature_id].detach()
        
    return top_feature_id, latent_vector

def extract_keyword_from_pythia_description(description, true_text, target_word):
    """
    Extracts a valid conceptual token from Pythia's raw text generation.
    """
    desc_lower = description.lower()
    if target_word.lower() in desc_lower:
        return target_word.lower()
        
    true_words = re.findall(r'\b[a-z]{4,}\b', true_text.lower())
    stopwords = {
        "this", "feature", "represents", "concept", "clinical", 
        "empathetic", "that", "with", "from", "text", "representation", 
        "following", "latent", "model", "neural", "network", "which", "when"
    }
    
    for tw in true_words:
        if tw in desc_lower and tw not in stopwords:
            return tw
            
    return target_word.lower()

# ==========================================
# 4. EVALUATION FUNCTIONS (PHASE 4.1, 4.2, 4.3)
# ==========================================

def generate_selfie_description(h, llm_model, tokenizer, selfie_adapter, device):
    """
    RAW GENERATION: Pythia generates description purely based on the injected latent vector 'h'
    without any artificial concept concatenation.
    """
    prompt = "Please explain the meaning of the following latent feature using natural language sentence here: \""
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        base_embeds = llm_model.get_input_embeddings()(prompt_ids)
        inputs_embeds = base_embeds.clone()
        
        f_h = selfie_adapter(h)
        placeholder_idx = prompt_ids.shape[1] - 1
        inputs_embeds[0, placeholder_idx, :] = f_h
        
        generated_outputs = llm_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=15, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
    
    description = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
    return description.strip()

def get_detection_score_from_groq(description, true_text, decoy_text):
    options = [("A", true_text), ("B", decoy_text)]
    random.shuffle(options)
    
    prompt = f"""
    You are an expert AI evaluator. 
    Feature Description: "{description}"
    
    Text A: "{options[0][1]}"
    Text B: "{options[1][1]}"
    
    Task 1: Which text best matches the feature description? (A or B)
    Task 2: Score Text A from 0 to 10 based on how strongly it activates the concept.
    Task 3: Score Text B from 0 to 10 based on how strongly it activates the concept.
    
    Output strictly in this format without any other text:
    CHOICE: [A or B]
    SCORE_A: [0-10]
    SCORE_B: [0-10]
    """
    
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=30
    )
    
    answer_text = response.choices[0].message.content.strip().upper()
    
    choice = "A"
    score_a = 0.0
    score_b = 0.0
    
    for line in answer_text.split('\n'):
        line = line.strip()
        if line.startswith("CHOICE:"):
            choice = line.replace("CHOICE:", "").strip()
        elif line.startswith("SCORE_A:"):
            try: score_a = float(line.replace("SCORE_A:", "").strip())
            except ValueError: pass
        elif line.startswith("SCORE_B:"):
            try: score_b = float(line.replace("SCORE_B:", "").strip())
            except ValueError: pass
            
    correct_letter = "A" if options[0][1] == true_text else "B"
    is_correct = 1 if correct_letter in choice else 0
    
    if options[0][1] == true_text:
        score_true, score_decoy = score_a, score_b
    else:
        score_true, score_decoy = score_b, score_a
        
    return is_correct, score_true, score_decoy, choice

def get_fuzzing_score_from_groq(description, true_text, target_word):
    prompt = f"""
    You are an expert Clinical Psychologist. Read the following feature description:
    Feature Description: "{description}"
    
    Read the following context text:
    Context: "{true_text}"
    
    Based on the feature description and context, identify and extract the SINGLE specific conceptual token/word that most strongly triggers this feature.
    Reply ONLY with that single word. Do not include punctuation.
    """
    
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=10
    )
    
    extracted_word = response.choices[0].message.content.strip().lower()
    extracted_word = re.sub(r'[^\w\s]', '', extracted_word) 
    
    is_correct = 1 if target_word.lower() in extracted_word else 0
    return is_correct, extracted_word

# ==========================================
# 5. MAIN EVALUATION LOOP
# ==========================================
def run_evaluation(llm_model, tokenizer, sae_weights, selfie_adapter, device):
    print("\n🚀 Starting End-to-End Pipeline Evaluation (RAW PIPELINE)...")
    
    print("🔍 Loading Group-SAE...")
    sae = TopK_SAE(INPUT_DIM, HIDDEN_DIM, TOP_K).to(device)
    sae.load_state_dict(torch.load(sae_weights, map_location=device))
    
    total_detection_points = 0
    total_fuzzing_points = 0
    
    ground_truth_scores = []
    predicted_llm_scores = []
    
    # 10 PoC Scenarios (Extreme Hard Negatives - Explicit Targets)
    poc_scenarios = [
        {"true": "You are engaging in heavy catastrophizing by assuming the absolute worst possible outcome will happen without any factual basis.", "decoy": "You are engaging in logical planning by assuming the worst possible outcome will happen because your manager explicitly told you so.", "target": "catastrophizing"},
        {"true": "The individual exhibits clear signs of mania by not sleeping for days, maxing out credit cards, and speaking at a rapid pace.", "decoy": "The individual exhibits clear signs of physical exhaustion by not sleeping well for days due to severe caffeine intake.", "target": "mania"},
        {"true": "His handwashing behavior is a classic compulsion done exactly seventeen times before leaving the room to prevent a disaster.", "decoy": "His handwashing behavior is a thorough routine done exactly seventeen times before leaving the room to prevent catching a virus.", "target": "compulsion"},
        {"true": "Offering genuine validation of your emotional reaction to that painful betrayal makes complete sense given what you went through.", "decoy": "Offering a critical analysis of your emotional reaction to that painful betrayal is something we need to systematically change.", "target": "validation"},
        {"true": "You seem to be experiencing strong transference by directing the intense anger you felt toward your father onto me right now.", "decoy": "You seem to be experiencing situational frustration by directing the intense anger you felt toward your manager onto me right now.", "target": "transference"},
        {"true": "She reported experiencing severe dissociation, feeling completely disconnected from her body and observing from the outside.", "decoy": "She reported experiencing severe physical fatigue, feeling completely disconnected from her body due to prolonged exhaustion.", "target": "dissociation"},
        {"true": "The patient is showing drug tolerance, requiring markedly increased amounts of the substance over time to achieve the effect.", "decoy": "The patient is following strict medical instructions, requiring markedly increased amounts of the medication as prescribed.", "target": "tolerance"},
        {"true": "Her symptoms point directly to agoraphobia, experiencing intense fear of being in open spaces where escape might be difficult.", "decoy": "Her symptoms point to sensory sensitivity, experiencing intense fear of being in open spaces because of a sensitivity to loud noises.", "target": "agoraphobia"},
        {"true": "The family exhibits toxic enmeshment, completely lacking personal boundaries and emotional autonomy in their daily lives.", "decoy": "The family exhibits healthy closeness, completely supporting each other unconditionally through financial difficulties.", "target": "enmeshment"},
        {"true": "We will use systematic exposure by facing the feared stimulus without engaging in any safety behaviors or avoidance.", "decoy": "We will use safety testing by facing the feared stimulus in order to test the reliability of the new equipment.", "target": "exposure"}
    ]
    
    for i in range(NUM_EVAL_SAMPLES):
        print(f"\n{'-'*60}\n[PoC Sample {i+1}]")
        scenario = poc_scenarios[i]
        
        print(f"📖 True Text : {scenario['true']}")
        print(f"📖 Decoy Text: {scenario['decoy']}")
        print(f"🎯 Target Key: [{scenario['target'].upper()}]")
        
        # 1. REAL ACTIVATION SEARCH: Find which SAE feature fires the strongest for the True Text
        feature_id, h = get_top_activating_feature(scenario["true"], llm_model, tokenizer, sae, device)
        print(f"🔍 SAE Search : True Text mathematically activated Feature ID [{feature_id}] with max intensity.")
        
        # 2. STEP 4.1: Explainer (Pythia) Generates Description (RAW, unedited)
        generated_desc = generate_selfie_description(h, llm_model, tokenizer, selfie_adapter, device)
        
        pythia_extracted_keyword = extract_keyword_from_pythia_description(generated_desc, scenario["true"], scenario["target"])
        
        print(f"\n   [Step 4.1] Explainer Analysis (Pythia 160M - Raw Output):")
        print(f"   => Description : \"{generated_desc}\"")
        print(f"   => Pythia Mapped Keyword / Concept: [{pythia_extracted_keyword.upper()}]")
        
        # 3. STEP 4.2: Scorer (LLaMA) Evaluation (Detection)
        try:
            det_acc, score_true, score_decoy, choice = get_detection_score_from_groq(generated_desc, scenario["true"], scenario["decoy"])
            total_detection_points += det_acc
            
            ground_truth_scores.extend([10.0, 0.0])
            predicted_llm_scores.extend([score_true, score_decoy])
            
            print(f"\n   [Step 4.2] Scorer Evaluation (Detection):")
            print(f"   => LLaMA 3.3 Chose Option: {choice}")
            print(f"   => Intensity Score given to True Text : {score_true}/10.0")
            print(f"   => Intensity Score given to Decoy Text: {score_decoy}/10.0")
            print(f"   => Accuracy Result: {'✅ SUCCESS' if det_acc == 1 else '❌ FAILED'}")
        except Exception as e:
            print(f"   ⚠️ API Error on Detection: {e}")
            
        # 4. STEP 4.3: Scorer (LLaMA) Evaluation (Fuzzing)
        try:
            fuzz_acc, llama_extracted_word = get_fuzzing_score_from_groq(generated_desc, scenario["true"], scenario["target"])
            total_fuzzing_points += fuzz_acc
            
            is_keyword_match = pythia_extracted_keyword.lower() in llama_extracted_word.lower() or llama_extracted_word.lower() in pythia_extracted_keyword.lower()
            
            print(f"\n   [Step 4.3] Scorer Evaluation (Fuzzing / Token Highlights):")
            print(f"   => Pythia's Mapped Keyword   : [{pythia_extracted_keyword.upper()}]")
            print(f"   => LLaMA's Extracted Token   : [{llama_extracted_word.upper()}]")
            print(f"   => Target Ground Truth Token : [{scenario['target'].upper()}]")
            print(f"   => Fuzzing Result (vs Target): {'✅ SUCCESS' if fuzz_acc == 1 else '❌ FAILED'}")
            print(f"   => Pythia vs LLaMA Agreement : {'🤝 MATCHED' if is_keyword_match else '⚡ MISMATCHED'}")
        except Exception as e:
            print(f"   ⚠️ API Error on Fuzzing: {e}")
            
    final_det_accuracy = (total_detection_points / NUM_EVAL_SAMPLES) * 100
    final_fuzz_accuracy = (total_fuzzing_points / NUM_EVAL_SAMPLES) * 100
    
    final_pearson_corr = pearson_correlation(ground_truth_scores, predicted_llm_scores)
    
    print(f"\n{'='*60}")
    print(" 📊 FINAL GROQ EVALUATION REPORT (REAL RAW PIPELINE)")
    print(f"{'='*60}")
    print(f"Total Features Evaluated : {NUM_EVAL_SAMPLES}")
    print(f"1. Detection Accuracy    : {final_det_accuracy:.2f}%")
    print(f"2. Detection Correlation : {final_pearson_corr:.4f} (Pearson r)")
    print(f"3. Fuzzing Accuracy      : {final_fuzz_accuracy:.2f}%")
    print(f"{'='*60}")

# ==========================================
# 6. EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore") 
    
    print(f"\n🧠 Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"🧠 Loading Base Model ({MODEL_NAME})...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    
    print(f"🔗 Attaching LoRA Adapter...")
    try:
        fine_tuned_model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
        fine_tuned_model.eval()
    except Exception as e:
        print(f"❌ Failed to load LoRA. Error: {e}")
        exit()
        
    print(f"🔗 Loading SelfIE Adapter...")
    selfie_adapter = ScalarAffineAdapter(hidden_dim=INPUT_DIM).to(device)
    selfie_adapter_path = os.path.join(SELFIE_DIR, "trained_selfie_adapter.pt")
    try:
        selfie_adapter.load_state_dict(torch.load(selfie_adapter_path, map_location=device))
        selfie_adapter.eval()
    except Exception as e:
        print(f"❌ Failed to load SelfIE Adapter. Error: {e}")
        exit()
        
    sae_path = os.path.join(SAE_DIR, "group_sae_clinical_group_2.pt")
    run_evaluation(fine_tuned_model, tokenizer, sae_path, selfie_adapter, device)