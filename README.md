# Medical Triage NLP Pipeline

**Automatic Medical Specialty Classification using Enhanced BiLSTM-MHA & Fine-Tuned BioBERT**

A deep learning-based NLP system that automatically predicts the appropriate medical specialty from clinical text descriptions, achieving **86.75% accuracy** using fine-tuned BioBERT.

---

## 📋 Overview

When patients arrive at a hospital, manual triage—deciding which specialist they should see—is slow, error-prone, and cognitively demanding. This project automates that process by training two state-of-the-art deep learning models to classify clinical notes into one of **6 medical specialties**:

- **Cardiology**
- **Gastroenterology**
- **Neurology**
- **Obstetrics / Gynecology**
- **Orthopedics**
- **Urology**

### Key Results

| Model | Test Accuracy | F1 Macro | Precision | Recall |
|-------|---------------|----------|-----------|--------|
| **Fine-tuned BioBERT** ⭐ | **86.75%** | **0.8678** | **0.8703** | **0.8675** |
| Enhanced BiLSTM-MHA | 73.90% | 0.7500 | 0.7699 | 0.7390 |
| TF-IDF + Logistic Regression | ~60–65% | — | — | — |

---

## 🏗️ Architecture

### Model 1: Enhanced BiLSTM with Multi-Head Self-Attention

A deep LSTM baseline that processes clinical text bidirectionally:

```
Input (300 tokens)
    ↓
Word Embedding (300-d)
    ↓
Bidirectional LSTM (2 layers, 384 hidden units each direction)
    ↓
Multi-Head Self-Attention (4 heads)
    ↓
Layer Normalization
    ↓
Classification Head (768 → 512 → 256 → 6 classes)
    ↓
Output (6 specialties)
```

**Why BiLSTM?** Clinical notes have important context on both sides of a word (e.g., "no chest pain" — "no" comes before "pain"). BiLSTM processes text in both directions simultaneously.

**Parameters:** 10,104,990

### Model 2: Fine-Tuned BioBERT with Dual Pooling

A transformer-based model pre-trained on 18 billion biomedical words:

```
Input (256 WordPiece tokens)
    ↓
BioBERT Encoder (12 transformer layers, 12 attention heads)
    ↓
[CLS] Token Extraction (768-d)        + Mean-Pooled Representations (768-d)
    ↓                                      ↓
    └──────────────────┬───────────────────┘
                       ↓
            Concatenation (1536-d)
                       ↓
            Classification Head (1536 → 512 → 256 → 6 classes)
                       ↓
                   Output (6 specialties)
```

**Why BioBERT?** Pre-trained on PubMed and PMC clinical literature, it already understands biomedical vocabulary.

**Key Innovation — Dual Pooling:** Instead of only using the [CLS] token, we concatenate both global sentence meaning ([CLS]) and local word-level signals (mean pooling) for richer representations.

**Parameters:** 109,234,182 (~11× larger than BiLSTM)

---

## 📊 Dataset

**Source:** [MTSamples Kaggle Dataset](https://www.kaggle.com/tboyle10/medicaltranscriptions) — 4,999 real clinical transcription notes

**Filtering Process:**

| Stage | Samples | Action |
|-------|---------|--------|
| Raw data | 4,999 | Original mtsamples.csv |
| After removing non-clinical rows | 2,390 | Removed discharge summaries, radiology reports, letters |
| After class-size filter (≥100 samples) | **1,660** | Kept only 6 specialties with sufficient data |

**Final Split (70% / 15% / 15%, Stratified):**

| Specialty | Total | Train | Val | Test |
|-----------|-------|-------|-----|------|
| Orthopedics | 423 | 296 | 63 | 64 |
| Cardiology | 372 | 260 | 56 | 56 |
| Neurology | 317 | 222 | 48 | 47 |
| Gastroenterology | 230 | 161 | 34 | 35 |
| Obstetrics / Gynecology | 160 | 112 | 24 | 24 |
| Urology | 158 | 111 | 24 | 23 |
| **TOTAL** | **1,660** | **1,162** | **249** | **249** |

---

## 🔧 Preprocessing Pipeline

Custom preprocessing specifically optimized for medical text (applied to BiLSTM-MHA path only):

1. **Lowercase** all text
2. **Remove dosage patterns:** '50mg', '2.5ml', '120mmHg' → removed via regex
3. **Remove standalone numbers:** '1.', '2.' list markers → removed
4. **Remove special characters:** commas, brackets, colons → removed
5. **Custom stopword removal:** Standard NLTK stopwords, but **preserve critical medical terms:**
   - 'no', 'not', 'without', 'right', 'left', 'upper', 'lower'
6. **Lemmatization:** 'fractures' → 'fracture', 'presenting' → 'present'

**Note:** BioBERT uses raw text directly (no preprocessing) because it has its own WordPiece tokenizer.

### Example:

**Before:**
```
"2-D M-MODE: ,1. Left atrial enlargement with left atrial diameter of 4.7 cm.,2. Normal 
size right and left ventricle.,3. Normal LV systolic function with left ventricular 
ejection fraction of 51%..."
```

**After:**
```
"mode left atrial enlargement with left atrial diameter normal size right left ventricle 
normal systolic function with left ventricular ejection fraction normal diastolic 
function pericardial effusion..."
```

---

## 🎯 Key Training Techniques

### BiLSTM-MHA

- **Loss:** Focal Loss (γ=2.0) — down-weights easy samples, focuses on hard cases
- **Class Weighting:** Squared-inverse frequency to handle class imbalance (296 Ortho vs. 111 Urology samples)
- **Learning Rate:** CosineAnnealingLR (3e-4 → 1e-5)
- **Regularization:** Dropout (0.35), Gradient Clipping (max_norm=1.0)
- **Early Stopping:** Patience=6, stopped at epoch 14

### BioBERT Fine-Tuning

- **Optimizer:** AdamW with weight-decay grouping (0.01 on weights, 0.0 on biases/LayerNorm)
- **Loss:** Cross-Entropy + Label Smoothing (0.05)
- **Learning Rate:** Cosine decay + linear warmup (146/1460 steps), init 2e-5
- **Regularization:** Dropout (0.25), full fine-tuning (no frozen layers)
- **Early Stopping:** Patience=3, stopped at epoch 8 (best model from epoch 5)

---

## 📈 Results & Analysis

### Per-Class Performance (BioBERT)

| Specialty | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Cardiology | 0.9464 | 0.9464 | 0.9464 |
| Gastroenterology | 0.8378 | 0.8857 | 0.8611 |
| Neurology | 0.8049 | 0.7021 | **0.7500** ⚠️ |
| Obs/Gynecology | 1.0000 | 0.7917 | 0.8837 |
| Orthopedics | 0.8382 | 0.8906 | 0.8636 |
| Urology | 0.8214 | 1.0000 | 0.9020 |

### Model Agreement Analysis

| Outcome | Count | Percentage | Interpretation |
|---------|-------|-----------|-----------------|
| Both models correct | 172 | 69.1% | High-confidence predictions |
| Only BioBERT correct | 44 | 17.7% | BioBERT's pre-training advantage |
| Only BiLSTM correct | 12 | 4.8% | Ensemble value — BiLSTM catches edge cases |
| Neither correct | 21 | 8.4% | Genuinely ambiguous cases (Neuro/Ortho overlap) |

### Most Common Error: Neurology ↔ Orthopedics

Both specialties present with overlapping symptoms:
- Radiating pain
- Motor weakness
- Reduced reflexes

Example confusion: A herniated disc (Orthopedics) vs. peripheral neuropathy (Neurology) can present with nearly identical clinical descriptions.

---

## 🧪 Real-Time Triage Demo

### Successfully Classified Cases:

| Input | Predicted | Confidence | Status |
|-------|-----------|------------|--------|
| Severe crushing chest pain, diaphoresis, nausea | Cardiology | 92.1% | ✅ |
| Sudden face drooping, right arm weakness, speech slurred | Neurology | 84.1% | ✅ |
| Knee locked in flexion after pivoting | Orthopedics | 91.8% | ✅ |
| Vomiting coffee-ground material, epigastric pain | Gastroenterology | 96.7% | ✅ |
| Severe loin-to-groin pain with haematuria | Urology | 83.1% | ✅ |

### Unseen Specialties (Keyword Override):

| Input | Predicted | Method | Status |
|-------|-----------|--------|--------|
| I feel completely hopeless, suicidal thoughts daily | Psychiatry | KEYWORD | ✅ |
| Leukemia, platelet 12, starting chemotherapy | Hematology | KEYWORD | ✅ |
| Blocked nose, ringing in ear, lost sense of smell | ENT | KEYWORD | ✅ |

### Failed Cases:

| Input | Expected | Predicted | Issue |
|-------|----------|-----------|-------|
| My 2-year-old baby, 40°C fever | Pediatrics | Obs/Gyn (46.8%) | Phrasing mismatch ("my baby" not in trigger) |
| Silvery scaly plaques on elbows, nail pitting | Dermatology | Orthopedics (91.7%) | Override didn't fire; model never trained on dermatology |

---

## 📦 Installation & Requirements

### Dependencies

```
PyTorch >= 2.1.0
transformers >= 4.41.0
scikit-learn
pandas
numpy
matplotlib
seaborn
nltk
accelerate
sentencepiece
```

### Install

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn nltk accelerate sentencepiece
```

### Download NLTK Data

```python
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
```

---

## 🚀 Usage

### 1. **Prepare Data**

Download the MTSamples dataset:
```bash
kaggle datasets download -d tboyle10/medicaltranscriptions
```

### 2. **Preprocess & Train BioBERT**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# Load pre-trained BioBERT
model_name = "dmis-lab/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=6
)

# Configure training
training_args = TrainingArguments(
    output_dir="./biobert_finetuned",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=146,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

### 3. **Make Predictions**

```python
def predict_specialty(clinical_text, model, tokenizer):
    inputs = tokenizer(
        clinical_text, 
        max_length=256, 
        truncation=True, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()
    
    specialties = [
        "Cardiology", "Gastroenterology", "Neurology",
        "Obstetrics/Gynecology", "Orthopedics", "Urology"
    ]
    
    return specialties[pred_class], confidence

# Example
text = "Severe crushing chest pain with diaphoresis"
specialty, conf = predict_specialty(text, model, tokenizer)
print(f"Predicted: {specialty} ({conf:.2%})")
```

---

## 🔍 Key Findings

### ✅ What Worked Well

1. **BioBERT Dominance:** +12.85% accuracy over BiLSTM-MHA due to biomedical pre-training
2. **Dual Pooling:** Combining [CLS] and mean-pooled representations improved classifier flexibility
3. **Focal Loss:** Effectively handled class imbalance, improved minority class performance
4. **High-Confidence Classes:** Cardiology (94.6% precision) and Urology (100% recall) are reliably classified
5. **Keyword Override:** Successfully catches obvious unseen specialties (Psychiatry, Hematology, ENT)

### ⚠️ Limitations & Failures

1. **Neurology-Orthopedics Confusion:** Overlapping symptoms make this the hardest classification (F1=0.75)
2. **BiLSTM Overfitting:** 94% training accuracy vs 73.9% test accuracy; 1,162 training samples insufficient for 10M parameters
3. **Keyword Override Failures:** Phrasing sensitivity ("my 2-year-old" vs "baby" triggers) and missing patterns
4. **CPU-Only Training:** Limited to small batch sizes (8) and sequence lengths (256), slow iteration

---

## 🔮 Future Work

1. **Expand to more specialties** using MIMIC-III (53,000+ patients, 50+ specialties)
2. **Try ClinicalBERT or BioMedLM** (GPT-based biomedical models) for potential accuracy gains
3. **Add structured features:** age, vital signs, lab values alongside free-text
4. **Improve keyword override** with regex patterns and contextual rules
5. **Data augmentation:** back-translation, synonym substitution for underrepresented classes
6. **GPU acceleration:** Enable larger batch sizes, longer sequences, faster iteration

---

## 📚 Related Work & References

This project builds on foundational work in clinical NLP:

1. **BERT** (Devlin et al., 2019) — Masked language modeling, transformers
2. **BioBERT** (Lee et al., 2020) — Biomedical pre-training on PubMed + PMC
3. **BiLSTM-CRF** (Huang et al., 2015) — Bidirectional sequence modeling
4. **Attention Mechanisms** (Bahdanau et al., 2015) — Learning to focus on relevant tokens
5. **ClinicalBERT** (Huang et al., 2019) — Domain-specific BERT for clinical notes
6. **MIMIC-III** (Johnson et al., 2016) — Clinical text preprocessing standards

See full references in the project report.

---

## 👥 Team

- **Divyansh Sharma** – 102303964
- **Deepinder Singh Saini** – 102303673
- **Anshdeep Handa** – 102303124

**Subgroup:** 3C62  
**Submitted To:** Dr. Kanupriya  
**Institution:** Thapar Institute of Engineering and Technology (TIET)

---

## 📝 License

This project is for educational purposes. The MTSamples dataset is publicly available on Kaggle.

---

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome. Please open an issue or submit a pull request.

---

## 💡 Quick Reference for Viva

| Question | Answer |
|----------|--------|
| Why BiLSTM and not simple LSTM? | BiLSTM reads text in both directions; clinical notes have context on both sides of key words |
| What is Focal Loss? | Modified Cross-Entropy that down-weights easy samples; forces focus on hard misclassified cases |
| Why multi-head attention? | 4 heads allow simultaneous attention to 4 different types of clinical features |
| What is fine-tuning? | Taking BioBERT's pre-trained weights and continuing training on our 6-class task |
| What is dual pooling? | Concatenating [CLS] (768-d) + mean-pooled representations (768-d) → 1536-d input to classifier |
| Why did BioBERT beat BiLSTM? | BioBERT pre-trained on 18B biomedical words; BiLSTM learns from scratch with only 1,162 samples |
| What is label smoothing? | Using soft labels (0.05/0.95) instead of hard (0/1) to prevent overconfidence |
| Why is Neurology the hardest? | Neurology & Orthopedics share overlapping symptoms: referred pain, motor weakness, reduced reflexes |
| What is early stopping? | Monitor validation loss; stop and restore best model if no improvement for 3–6 epochs |
| Why GELU vs ReLU? | GELU is smooth near zero; ReLU hard-gates negatives to zero, losing gradient information |

---

## 📞 Contact & Questions

For questions about this project, contact the team or open an issue on the repository.

**Last Updated:** April 2024
