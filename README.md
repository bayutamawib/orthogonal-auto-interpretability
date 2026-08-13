# 🧠 Mind-Reading LLMs: Disentangling Empathy and Clinical Reasoning via Orthogonal Auto-Interpretability

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9DC3E.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Abstract:** This project presents an end-to-end Auto-Interpretability pipeline that addresses the "alignment tax" in Large Language Models (LLMs)—where training for empathy degrades objective clinical reasoning. By combining **Group-SAE**, **Subspace Orthogonalization**, and **Trained Self-Interpretation (SelfIE)**, this architecture not only extracts latent features but mathematically disentangles them, enabling the LLM to translate its own internal representations into natural language *without ever reading the source text*.

---

## 🚀 Core Technologies

* **Base Model:** `EleutherAI/pythia-160m`
* **Evaluation Model (Scorer):**
  * This Proof of Concept is using `Llama-3.1-8b-instant` (via Groq).
  * Further, we will be using `LLaMA 3.3 70B` (via Groq) for a scaled project.
* **Techniques:** LoRA, Group-SAE, SelfIE Adapter, Orthogonal Latent Projection

---

## 🔍 Repository Structure

```bash
├── data/
│   └── raw/
│       ├── ESConv.json                  # Empathy corpus
│       └── heliosbrahma_dataset.json    # Clinical psychology corpus
│
├── models/
│   ├── lora_adapter/                    # Bobot hasil Phase 1
│   ├── group_sae/                       # Bobot SAE hasil Phase 2
│   └── selfie_adapter/                  # Bobot SelfIE hasil Phase 3
│
├── src/
│   ├── phase1_train_lora.py             # Script LoRA Fine-Tuning Pythia-160M
│   ├── phase2_train_sae.py              # Script Group-SAE & Orthogonalization
│   ├── phase3_train_selfie.py           # Script pelatihan Trained SelfIE
│   ├── phase4_eval_scorer.py            # Script Detection & Fuzzing Scores
│   └── utils/
│
├── notebooks/
│   └── poc_pipeline.ipynb               # Kaggle Notebook untuk Proof of Concept
│
├── requirements.txt                     # Dependencies (torch, transformers, peft, dll)
└── README.md                            # Dokumentasi Proyek
```

---

## 🏗️ Architecture & Methodology Workflow

Our pipeline maps the journey from a raw base model to a fully interpretable, self-explaining neural network.

### 🔹 Phase 1: Base Model Fine-Tuning

We fine-tuned **Pythia-160M** using **LoRA** on three specialized datasets:

* **1 Empathy Dataset:** `ESConv`
* **2 Clinical Datasets:** `Heliosbrahma` & `HF Mental Health Condition Classification`

This step anchors the model's domain understanding firmly within psychological and clinical contexts.

### 🔹 Phase 2: Feature Extraction (Group-SAE)

Building on the efficiency of **Group-SAE** (*Ghilardi et al., 2025*), we extract the latent feature vector ($h$) that fires mathematically when the base model processes specific clinical texts. We group adjacent layers to make extraction computationally efficient while preserving high-level semantic features.

### 🔹 Phase 3: Orthogonalization (The Key Innovation)

Inspired by the discovery that empathy can induce sycophancy (*Ibrahim et al., 2025*), and leveraging Sparse Activation Editing (*Zhao et al., 2025*), we implemented a novel **orthogonalization** technique.

We mathematically force the feature representing **Clinical Accuracy** (facts/diagnosis) to be orthogonal (perpendicular and independent) to the feature representing **Empathy** (tone/validation). This disentanglement ensures that when the model evaluates a diagnosis, the diagnostic signal remains completely untangled from emotional bias.

### 🔹 Phase 4: SelfIE Adapter & Zero-Shot Vector Translation

Using the Trained Self-Interpretation methodology (*Pepper et al., 2026*), we injected the untangled feature vector ($h$) back into Pythia's embedding space via a lightweight **SelfIE Adapter** (a scalar-affine transformation: $f_{adapter}(h) = (s \cdot h) + b$).

This acts as a mathematical bridge, enabling Pythia to zero-shot generate a natural language description of the active concept—**without the model ever seeing the original input text.**

### 🔹 Phase 5: Automated Evaluation (Explainer & Scorer)

We employ **LLaMA 3.3 70B** as an independent, blind Judge (Scorer) to validate the pipeline:

1. **Detection:** LLaMA reads Pythia's generated description and must distinguish between a *True Text* (which actually triggers the concept) and a *Decoy Text* (an Extreme Hard Negative).
2. **Fuzzing:** LLaMA reads Pythia's description and extracts the single most triggering token/word from the True Text.

---

## 📊 Proof of Concept (PoC) Results

Our initial Proof of Concept yielded highly promising results in zero-shot interpretability:

* 🎯 **Detection Accuracy:** `90.00%`
* 📈 **Detection Correlation:** `0.7216` (Strong Pearson *r*)
* 🔍 **Fuzzing Accuracy:** `100.00%`

### 💡 Technical Insight: "The Deductive Rescue"

During the PoC, the adapter was trained on a highly limited dataset (~500 samples), which caused Pythia to exhibit signs of catastrophic forgetting. It generated high-level domain clues mixed with hallucinated medical citations (e.g., *"Concept representation in clinical examination (Gulliver..."*).

**However, LLaMA achieved 100% fuzzing accuracy through contextual deduction.** Because the orthogonalized signal was inherently clean, LLaMA successfully used Pythia's broad "clinical" clue to deduce the exact triggering clinical word (e.g., "compulsion") from the source text. This proves that the architecture successfully transfers latent intent, even if the explainer's resolution requires a larger dataset to eliminate pre-training artifacts.

---

## 🗺️ Future Roadmap

* **Scale Adapter Training:** Increase the SelfIE adapter training data to 10,000+ samples to eliminate pre-training artifacts and improve description resolution.
* **Refine Layer Groups:** Focus Group-SAE extraction strictly on late layers (e.g., Layers 8–11 in Pythia-160M) to isolate high-level semantic concepts.

---

## 💻 Installation

```bash
git clone https://github.com/yourusername/ortho-groupsae-selfie-poc.git
cd ortho-groupsae-selfie-poc
pip install -r requirements.txt
```

*(Note: Groq API Key is required for the Scorer/Explainer module. Ensure `GROQ_API_KEY` is set in your environment).*

## 🚀 Usage

Execute the pipeline sequentially from the `src/` directory:

```bash
# 1. Fine-tune the base model
python src/phase1_train_lora.py

# 2. Extract features and enforce Orthogonalization
python src/phase2_train_sae.py

# 3. Train the SelfIE Adapter
python src/phase3_train_selfie.py

# 4. Run Automated Evaluation Pipeline
python src/phase4_eval_scorer.py
```

## 📚 References & Citation

This project builds upon the foundational research presented in the following papers:

* **Ghilardi, D., Belotti, F., Molinari, M., Ma, T., & Palmonari, M. (2025).** Group-SAE: Efficient Training of Sparse Autoencoders for Large Language Models via Layer Groups. *EMNLP 2025*.
* **Ibrahim, L., Hafner, F. S., & Rocher, L. (2025).** Training language models to be warm and empathetic makes them less reliable and more sycophantic. *arXiv preprint*.
* **Pepper, K., et al. (2026).** Learning Self-Interpretation from Interpretability Artifacts: Training Lightweight Adapters on Vector-Label Pairs. *arXiv preprint arXiv:2602.10352*.
* **Zhao, R., et al. (2025).** Sparse activation editing for reliable instruction following in narratives. *EMNLP 2025*.

If you use this code in your research, please cite this repository.
