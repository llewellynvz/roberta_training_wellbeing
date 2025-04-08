import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
import torch
import numpy as np
import joblib
from sklearn.utils import resample
import torch.nn.functional as F



# Step 1: Load CSV into a Pandas DataFrame
CSV_PATH = "train.xlsx"
df = pd.read_excel(CSV_PATH, engine="openpyxl")

# Step 2: Encode labels (convert text labels to integers)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])
label_list = list(label_encoder.classes_)  # ['acculturation stress', ...]

# Manual subcategory → domain mapping
subcategory_to_domain = {
    "acculturation_stress": "life_demands",
    "aging": "life_demands",
    "bullying": "life_demands",
    #........  
    "strengths_use_&_awareness": "wellbeing",
    "building_positive_relationships": "wellbeing",
    "work-life_balance": "life_resources",
}

# Step 2.5: Balance dataset by resampling minority classes (bottom X%)
def smart_resample(df, label_col="label", bottom_percentile=45, multiplier=1.8):
    label_counts = df[label_col].value_counts()
    threshold = np.percentile(label_counts, bottom_percentile)

    underrepresented_labels = label_counts[label_counts <= threshold].index.tolist()
    print(f"Resampling {len(underrepresented_labels)} classes below {bottom_percentile}th percentile (≤ {int(threshold)} samples).")

    df_resampled = df.copy()
    for label in underrepresented_labels:
        df_subset = df[df[label_col] == label]
        n_samples = int(multiplier * threshold)
        if len(df_subset) < n_samples:
            df_upsampled = resample(
                df_subset,
                replace=True,
                n_samples=n_samples,
                random_state=42
            )
            df_resampled = pd.concat([df_resampled, df_upsampled], axis=0)

    return df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle


df = smart_resample(df, label_col="label", bottom_percentile=45, multiplier=1.8)


# Step 3: Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)  # 80/20 train/test split
val_test_split = dataset["test"].train_test_split(test_size=0.5, seed=42)  # 50/50 for val/test
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": val_test_split["train"],
    "test": val_test_split["test"],
})

# Step 4: Tokenization
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def tokenize(batch):
    texts = [str(x) for x in batch["text"]]  # ensure list of strings
    return tokenizer(texts, padding=True, truncation=True, max_length=512)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    logits, labels = pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)

    # Log missing predictions
    missing_preds = set(np.unique(labels)) - set(np.unique(preds))
    if missing_preds:
        print(f"⚠️ Model never predicted these labels (indices): {missing_preds}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

dataset = dataset.map(tokenize, batched=True)

# Step 5: Set format
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Step 6: Model Config
id2label = {int(i): str(label) for i, label in enumerate(label_list)}
label2id = {str(label): int(i) for i, label in enumerate(label_list)}

config = AutoConfig.from_pretrained(
    "roberta-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    problem_type="single_label_classification"
)
#Sanity Check
#print("id2label:", id2label)
#print("label2id:", label2id)
#print("Label classes:", label_list)

# Step 7: Load model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=config)

# Step 8: Training args
training_args = TrainingArguments(
    output_dir="ROBERTA_MODEL_SINGLE",
    num_train_epochs=21,                    #Slightly more epochs for deeper learning
    per_device_train_batch_size=30,         #Smaller batch size allows for better generalization (less memory stress, bette ation)
    per_device_eval_batch_size=30,          #Match eval batch size with training for consistency
    evaluation_strategy="epoch",            #More frequent evaluation
    eval_steps=200,                         #Evaluate model frequently (adjust depending on your dataset size)
    logging_dir="ROBERTA_MODEL_SINGLE/logs",#Save Location
    logging_strategy="epoch",
    #logging_steps=200,
    learning_rate=2e-5,                     #Lower learning rate improves fine-tuning stability
    weight_decay=0.02,                      #Sliglty higher decay/regularization to reduce overfitting
    warmup_steps=1850,                      #Set to 10% of total steps for stable training
    save_strategy="epoch",                  #Save more frequently to catch best performing models
    #save_steps=200,
    load_best_model_at_end=True,
    save_total_limit=15    ,                 #Retain 15 top-performing checkpoints
    #report_to="tensorboard",                #Optional but highly recommended for in-depth performance analysis
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    lr_scheduler_type="cosine",              #Use cosine decay instead of default linear
    label_smoothing_factor=0.08,             #To reduce overfitting
    fp16=True,
)


# Step 9: Trainer
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
)



# Step 10: Train and Evaluate
trainer.train()
final_eval = trainer.evaluate()
print(" Final validation metrics:", final_eval)
trainer.evaluate()
# Evaluate on validation set again (optional)
test_metrics = trainer.evaluate(eval_dataset=dataset["test"])
print(" Final test set evaluation:", test_metrics)



# Optional: Save tokenizer and label encoder mappings
tokenizer.save_pretrained("ROBERTA_MODEL_SINGLE/FINAL_MODEL")
model.save_pretrained("ROBERTA_MODEL_SINGLE/FINAL_MODEL")
joblib.dump(label_encoder, "ROBERTA_MODEL_SINGLE/FINAL_MODEL/label_encoder.pkl")
logs = trainer.state.log_history
pd.DataFrame(logs).to_csv("ROBERTA_MODEL_SINGLE/FINAL_MODEL/training_logs.csv", index=False)


# Save label classes
import json
with open("ROBERTA_MODEL_SINGLE/FINAL_MODEL/label_map.json", "w") as f:
    json.dump(id2label, f)

from torch.utils.data import DataLoader, TensorDataset

def test_predictions_on_csv(csv_path, model, tokenizer, label_encoder, batch_size=16):
    df_test = pd.read_excel(csv_path, engine="openpyxl").dropna(subset=["text", "label"])
    texts = df_test["text"].tolist()

    # Encode labels
    df_test["label_encoded"] = label_encoder.transform(df_test["label"])

    # Tokenize
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Prepare DataLoader
    test_dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"]
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Predict
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)  # 🔧 Convert logits to probabilities
            all_logits.append(probs.cpu())
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())

    all_logits = torch.cat(all_logits).numpy()
    predicted_labels = label_encoder.inverse_transform(all_preds)

    # Add predicted domain column
    df_test["predicted_label"] = predicted_labels
    df_test["predicted_domain"] = df_test["predicted_label"].map(subcategory_to_domain).fillna("unknown")  #
    df_test["true_domain"] = df_test["label"].map(subcategory_to_domain).fillna("unknown")  #
    df_test["match"] = df_test["label"] == df_test["predicted_label"]

    # Accuracy
    accuracy = df_test["match"].mean()
    print(f"\n Test Prediction Accuracy on CSV: {accuracy:.2%}")

    # Preview mismatches
    potential_issues = df_test[~df_test["match"]]
    print(f"⚠️  Potential label issues: {len(potential_issues)}")
    print(potential_issues[["text", "label", "predicted_label", "predicted_domain"]].head())  #

    # Save results
    df_test.to_excel("ROBERTA_MODEL_SINGLE/predictions.xlsx", index=False)
    print("✅ Saved predictions to ROBERTA_MODEL_SINGLE/predictions.csv")

    # Metrics
    true = df_test["label_encoded"].to_numpy()
    pred = label_encoder.transform(df_test["predicted_label"])

    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        true, pred, average=None, zero_division=0
    )

    index_to_label = dict(enumerate(label_encoder.classes_))
    print("\n Per-class metrics (only for seen classes):")
    for i in range(len(class_precision)):
        label = index_to_label.get(i, f"Label {i}")
        print(f"{label} — P: {class_precision[i]:.2f}  R: {class_recall[i]:.2f}  F1: {class_f1[i]:.2f}")

    # Overall metrics
    metrics = compute_metrics((all_logits, true))
    print("\n Final test metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    missing_preds = set(np.unique(true)) - set(np.unique(pred))
    if missing_preds:
        print(f"⚠️ Model never predicted these labels (indices): {missing_preds}")

    per_class_data = [
        {"label": index_to_label[i], "precision": class_precision[i], "recall": class_recall[i], "f1": class_f1[i]}
        for i in range(len(class_precision))
    ]
    metrics_df = pd.DataFrame(per_class_data)
    metrics_df.to_excel("ROBERTA_MODEL_SINGLE/per_class_metrics.xlsx", index=False)

    return df_test

df_test = test_predictions_on_csv("train.xlsx", model, tokenizer, label_encoder)

def predict_texts(text_list, model, tokenizer, label_encoder, threshold=0.5):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encodings = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}  # Move tensors to GPU/CPU

    with torch.no_grad():
        outputs = model(**encodings)
        probs = F.softmax(outputs.logits, dim=1)  # Convert logits to probabilities
        predicted_classes = torch.argmax(probs, dim=1).cpu().numpy()
        confidences = torch.max(probs, dim=1).values.cpu().numpy()

    predicted_labels = label_encoder.inverse_transform(predicted_classes)

    for text, label, confidence in zip(text_list, predicted_labels, confidences):
        if confidence < threshold:
            label = "uncertain"
        print(f"\n📝 Text: {text}\nPredicted Label: {label} ({confidence:.2%} confidence)")


new_texts = [
    "I feel so out of place here. No one understands my customs.",
    "My colleagues laughed when I said I don’t drink alcohol.",
    "Im so greatful that I have a family that cares for me",
    "I like to do work tasks where i can use the things im good at",
]
predict_texts(new_texts, model, tokenizer, label_encoder)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix_from_predictions(df_test, output_path="ROBERTA_MODEL_SINGLE/confusion_matrix.png"):
    print("\nCreating confusion matrix from predicted and true labels...")

    # Ensure we have the predicted and true labels
    true_labels = df_test["label"]
    predicted_labels = df_test["predicted_label"]
    all_classes = sorted(list(set(true_labels) | set(predicted_labels)))

    # Generate the matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_classes)

    # Convert to DataFrame for readability
    cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)

    # Save matrix to Excel
    cm_df.to_excel("ROBERTA_MODEL_SINGLE/confusion_matrix_table.xlsx")
    print("Saved confusion matrix table to Excel.")

    # Plot using seaborn
    plt.figure(figsize=(22, 20))
    sns.heatmap(cm_df, cmap="Blues", square=True, cbar=True, xticklabels=True, yticklabels=True)
    plt.title("Confusion Matrix of True vs Predicted Labels", fontsize=16)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {output_path}")

plot_confusion_matrix_from_predictions(df_test)
