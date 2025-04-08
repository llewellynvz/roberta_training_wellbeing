# RoBERTa-Based Wellbeing Text Classifier Fine Tuning Model and Prediction Script

This repository contains the first step in full training and evaluation pipeline for a **RoBERTa-based text classifier** on wellbeing-related qualitative responses. It’s designed for structured classification of psychological themes from open-text input, with robust handling of real-world challenges like class imbalance, noisy labels, and domain mapping. It supports both **model training** and **batch inference across multiple survey columns**

Developed for organizational wellbeing diagnostics, research, and large-scale survey analysis.

---

## 🧠 What This Project Does

We take 600 000 raw wellbeing-related text data (from open-ended survey responses, reddit, twitter and quora) and classify each response into one of 200 predefined and manually classified subcategories (e.g., *work-life balance*, *social wellbeing*, *acculturation stress*, *autonomy* etc). Each subcategory can also be mapped to a broader domain (e.g., *life_demands*, *life_resources*, *personal_demands*, *personal_resources*, **wellbeing*), supporting more complex downstream analysis like multi-class classification.

In practical terms, it transforms free-text responses like *"I feel overwhelmed with work"* or *"I’m grateful for my friends"* into structured wellbeing subcategories such as:

- `burnout`
- `social_support`
- `identity_conflict`
- `workload_pressure`

These subcategories are also mapped to broader wellbeing **domains** (e.g., `life_demands`, `personal_resources`, `wellbeing`), enabling high-level reporting, modeling, or dashboarding.


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

## 🚀 How to Train the Model

### 📄 Input Format

`train.xlsx` must have two columns:

- `text`: open-ended qualitative response
- `label`: manually annotated subcategory label

### 🧪 Run Training

```bash
python train.py
```

This will:
- Upsample rare classes (bottom 45%) using smart resampling
- Train `roberta-base` with label smoothing, early stopping, and cosine LR scheduling
- Evaluate across accuracy, precision, recall, and F1
- Save the final model and label mappings

### 🧾 Output Files

After training, results are saved to:

```bash
ROBERTA_MODEL_SINGLE/FINAL_MODEL/
├── model + tokenizer
├── label_encoder.pkl
├── label_map.json
├── training_logs.csv
├── predictions.xlsx
└── per_class_metrics.xlsx
```

---

## 🧠 How to Predict from Raw Text Columns

Use the included `2. TEST PREDICTION.py` to generate structured predictions **across multiple text columns** in your dataset (e.g., from a wellbeing survey).

### 🧾 Input Format

Provide an Excel file (e.g., `test_predict2.xlsx`) with open-text columns like:

```python
["LifeDema", "LifeReso", "PersReso", "WellBe"]
```

Each row should represent a single participant or case.

---

### 🧪 Run Column-wise Prediction

```bash
python "predict.py"
```

This script will:
- Load your trained RoBERTa model and tokenizer
- Predict wellbeing subcategories per column using softmax probabilities
- Filter results using a confidence threshold (default: `0.55`)
- Output predictions and confidence scores to `Wellbeing_2025_Predictions_ByColumn.xlsx`

---

### ✅ Example Output

| Personal Demands                           | Predicted_Personal_Demands            | Confidence |
|-------------------------------------|--------------------------------|---------------------|
| "I’m exhausted all the time..."     | burnout, workload_pressure     | 0.88, 0.59          |
| "I feel like I don’t belong here."  | identity_conflict              | 0.73                |

---

## ⚙️ Adjusting Threshold

To make the model more strict or more exploratory, adjust the confidence threshold in the script:

```python
predict_column_multilabel(df, column_name, confidence_threshold=0.55)
```

---

## 📚 Label Schema and Mapping

Each subcategory label is optionally mapped to a **broader psychological domain**, e.g.:

```python
"burnout"              → "personal_demands"
"social_support"       → "life_resources"
"identity_conflict"    → "personal_demands"
"self_compassion"      → "wellbeing"
```

This enables both granular and thematic analysis.

---

## 👥 Use Cases

- Organizational diagnostics and reporting
- Mental health research
- Wellbeing survey analysis at scale
- Qualitative to quantitative conversion for SEM and modeling

---

## 📌 Notes

- This classifier uses **single-label prediction** per response as basis for a multi-label classification model 
- Ensure all required model files (model, tokenizer, label encoder) are saved before running predictions
- Columns with empty or missing text are skipped gracefully

---

## 👤 Author

This project was developed by an [Prof. Llewellyn van Zyl (PhD)](https://www.linkedin.com/in/llewellynvz), an organizational psychologist and data scientist specializing in human wellbeing, applied NLP, and data-sciences.


## 📬 Contact
💬 **[Open an Issue](https://github.com/llewellynvz/roberta_training_wellbeing)** on GitHub  
✉️ **Contact me via my website** [psynalytics.com](https://www.psynalytics.com)  

Feel free to fork, modify, or reach out if you'd like to contribute or collaborate on wellbeing-related ML pipelines.

---
