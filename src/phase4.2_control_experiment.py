import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from groq import Groq

# ==========================================
# 1. CONFIGURATION & INITIALIZATION
# ==========================================
# Ensure you have set your GROQ_API_KEY in the Kaggle environment/secrets
client = Groq(api_key="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")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

MODEL_NAME = "EleutherAI/pythia-160m"

print("🧠 Loading Base Model (Un-finetuned) and Tokenizer to GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
llm.eval()

# ==========================================
# 2. DATASETS (10 HARD NEGATIVE SCENARIOS)
# ==========================================
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

# ==========================================
# 3. GENERATION FUNCTION FOR BASE PYTHIA
# ==========================================
def get_base_pythia_guidance(text):
    # Prompting the base model to continue the text
    prompt = f"Text: {text}\nThis text describes the clinical concept of"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = llm.generate(
            **inputs, 
            max_new_tokens=15, 
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True
        )
    
    # Extract only the newly generated tokens
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    guidance = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    if not guidance:
        guidance = "[Model generated empty sequence]"
        
    return guidance

# ==========================================
# 4. EXPERIMENT EXECUTION
# ==========================================
def run_control_judge():
    print("\n⚖️ Starting Control Experiment: LLaMA Guided by BASE (Un-finetuned) Pythia...\n")
    correct_guesses = 0
    total = len(poc_scenarios)

    for i, scenario in enumerate(poc_scenarios):
        # 1. Get raw/hallucinated guidance from Base Pythia
        base_guidance = get_base_pythia_guidance(scenario['true'])
        
        # 2. Feed this raw guidance to LLaMA Judge
        prompt = f"""
        You are a clinical NLP judge. I will give you two texts: Text A and Text B.
        One is an accurate clinical/psychological text (True Text), and the other is a casual or logical text (Decoy Text).
        
        An explainer AI model analyzed the target concept and provided this guidance:
        "{base_guidance}"
        
        Text A: {scenario['true']}
        Text B: {scenario['decoy']}
        
        Based heavily on the explainer AI's guidance, which text represents the core concept of "{scenario['target']}"?
        Reply strictly with 'Text A' or 'Text B'.
        """

        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile", 
                temperature=0.0, 
            )
            
            answer = chat_completion.choices[0].message.content.strip()
            
            print(f"Scenario {i+1} (Target: {scenario['target']}):")
            print(f"🤖 Base Pythia Guidance : \"{base_guidance}\"")
            print(f"⚖️ LLaMA Answer         : {answer}")
            
            if "Text A" in answer:
                correct_guesses += 1
                print("Status: ✅ CORRECT")
            else:
                print("Status: ❌ INCORRECT")
                
        except Exception as e:
            print(f"An error occurred in scenario {i+1}: {e}")
            
        print("-" * 60)

    # 5. RESULTS & ANALYSIS
    accuracy = (correct_guesses / total) * 100
    
    print("\n" + "="*60)
    print("📊 BASELINE CONTROL RESULTS (Base Pythia vs Fine-Tuned Pythia)")
    print("="*60)
    print(f"Accuracy with Base Pythia Guidance: {correct_guesses}/{total} ({accuracy:.2f}%)")
    print("="*60)
    
    print("\n💡 HOW TO USE THIS IN YOUR RESEARCH:")
    print("- Qualitative Proof: Look at the 'Base Pythia Guidance' prints above. They are likely gibberish,")
    print("  hallucinations, or basic text continuations. Compare this to the semantic descriptions generated")
    print("  by your SelfIE Fine-Tuned model in Phase 1.")
    print("- Quantitative Proof: If this accuracy is lower than your PoC, it proves your SelfIE adapter")
    print("  actively enhances downstream interpretability. If it's still 100%, it provides a solid foundation")
    print("  to discuss 'Judge Over-Reliance' or 'Evaluation Saturation' in your paper.")

if __name__ == "__main__":
    run_control_judge()