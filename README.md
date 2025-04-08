# RoBERTa-Based Wellbeing Text Classifier Fine Tuning Model

This repository contains the first step in full training and evaluation pipeline for a **RoBERTa-based text classifier** on wellbeing-related qualitative responses. It’s designed for structured classification of psychological themes from open-text input, with robust handling of real-world challenges like class imbalance, noisy labels, and domain mapping.

---

## 🧠 What This Project Does

We take 600 000 raw wellbeing-related text data (from open-ended survey responses, reddit, twitter and quora) and classify each response into one of 200 predefined and manually classified subcategories (e.g., *work-life balance*, *social wellbeing*, *acculturation stress*, *autonomy* etc). Each subcategory can also be mapped to a broader domain (e.g., *life_demands*, *life_resources*, *personal_demands*, *personal_resources*, **wellbeing*), supporting more complex downstream analysis like multi-class classification.

### Key Capabilities:
- Fine-tunes `roberta-base` for **single-label classification**
- Uses **stratified upsampling** to rebalance underrepresented subcategories
- Tracks precision, recall, and F1 across all classes
- Maps predicted subcategories to broader psychological **domains**
- Outputs results in Excel for easy integration into quantitative pipelines

---

## 🔧 Project Structure

- `train.xlsx`: Input file with three columns inline with roberta dataset requirements — `text` (quoted examples) and `label` (manual classified categories) and `domain` (higher order class domain)
- `train.py`: Main training script with tokenization, model setup, smart resampling, and evaluation
- `predictions.xlsx`: Output file containing predicted labels and domains
- `ROBERTA_MODEL_SINGLE/`: Folder with trained model, tokenizer, logs, and label mappings

---

## 📦 Dependencies

Install using pip:

```bash
pip install pandas scikit-learn transformers datasets openpyxl torch joblib numpy
