# SURE-Session-Recommendation

Implementation and extension of:

**SURE: Session-Based Uninteresting Item Removal for Enhanced Recommendations**

---

## 📌 Project Overview

This project implements a lightweight version of the SURE framework for sequential recommendation by:

- Identifying **uninteresting (negative) items**
- Applying **Association Rule Mining (ARM)** to detect noise
- Filtering user interaction sequences
- Improving recommendation quality using **sequence modeling**

### 🔬 Extensions Beyond Paper

- Comparison of **Apriori vs FP-Growth**
- Addition of **Short Sequence Enhancement (SSE)**
- Lightweight **Sequential Pattern Mining (SPM-lite)**
- **Runtime + Efficiency Analysis across hardware**
- Trade-off analysis between **accuracy vs filtering strength**

---

## ⚙️ Methodology

### 1. Uninteresting Item Identification
- Ratings **< 3** are treated as negative feedback
- Used to build transactions for ARM

### 2. Association Rule Mining (ARM)
- Algorithms:
  - Apriori
  - FP-Growth
- Extracts frequently co-occurring **uninteresting items**
- Used to filter noisy items from sequences

### 3. Sequential Pattern Mining (SPM-lite)
- Lightweight bigram-based pattern extraction
- Captures **ordered negative transitions (i → j)**
- Complements ARM by modeling temporal dependencies

### 4. Short Sequence Enhancement (SSE)
- Reverse Markov model trained on long sequences
- Generates prefix items for short users
- Addresses **data sparsity**

### 5. Recommendation Model
- First-order **Markov (Bigram) Model**
- Predicts next item based on last interaction

### 6. Evaluation
- Metrics:
  - MRR (Mean Reciprocal Rank)
  - nDCG (Normalized Discounted Cumulative Gain)
- Leave-one-out split:
  - Last item → test
  - Remaining → train

---

### 🔍 Key Insights

- ARM improves recommendation by removing noisy patterns
- SSE improves performance for short sequences
- SPM improves **ranking quality (nDCG)** but slightly affects top-1 prediction (MRR)
- Excessive filtering degrades performance → **controlled filtering is critical**

---

## ⚡ Efficiency Analysis

- Reverse model (SSE): **< 0.1 sec (negligible)**
- ARM mining: **dominant cost (~2–4 sec)**
- FP-Growth vs Apriori:
  - Performance varies with hardware and dataset sparsity

---

## 🧠 Assumptions & Design Choices

To ensure feasibility and efficiency:

- Transformer models (SASRec) are approximated using **Markov models**
- Reverse Transformer → replaced with **Reverse Markov model**
- Full SPM → approximated with **bigram-based SPM-lite**
- Dataset used:
  - MovieLens 100K (explicit ratings, converted to implicit feedback)

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
````

---

## 📌 Conclusion

This work demonstrates that:

* Combining **ARM + SSE + SPM-lite** improves recommendation quality
* Lightweight probabilistic models can approximate complex architectures
* Proper filtering is crucial — both under-filtering and over-filtering impact performance

---

## 📚 Reference Base Paper
N. Sivakumar, A. Motha, G. Suganeshwari, S. P. Syed Ibrahim and V. Sugumaran, "SURE: Session-Based Uninteresting Item Removal for Enhanced Recommendations," in IEEE Access, vol. 13, pp. 43904-43918, 2025, doi: 10.1109/ACCESS.2025.3549133

