import pandas as pd
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import joblib
import numpy as np

# === Load model, tokenizer, and label encoder ===
model_path = "ROBERTA_MODEL_SINGLE/FINAL_MODEL" 

model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")

# === Device config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Load test data ===
df = pd.read_excel("test_predict2.xlsx")

# === Function to predict top-k categories for a given column ===
def predict_column_multilabel(df, column_name, confidence_threshold=0.55):
    df_valid = df[df[column_name].notna() & (df[column_name].str.strip() != "")].copy()
    df_valid[f"{column_name}_row_index"] = df_valid.index
    texts = df_valid[column_name].astype(str).tolist()

    predicted_labels = []
    predicted_confidences = []

    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()

        for prob_row in probs:
            above_thresh = prob_row >= confidence_threshold
            labels = label_encoder.inverse_transform(np.where(above_thresh)[0])
            scores = prob_row[above_thresh]

            if len(labels) == 0:
                predicted_labels.append("")  # nothing confidently predicted
                predicted_confidences.append("")
            else:
                predicted_labels.append(", ".join(str(label) for label in labels))
                predicted_confidences.append(", ".join(f"{s:.2f}" for s in scores))

    df.loc[df_valid.index, f"Predicted_{column_name}"] = predicted_labels
    df.loc[df_valid.index, f"Confidence_{column_name}"] = predicted_confidences

    return df


# === Apply predictions to each relevant column ===
for col in ["LifeDema", "LifeReso", "PersReso", "WellBe"]:
    df = predict_column_multilabel(df, col, confidence_threshold=0.55)

# === Save results ===
df.to_excel("predictions.xlsx", index=False)
print("✅ Predictions saved to output file")
