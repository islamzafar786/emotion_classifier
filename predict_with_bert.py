import os
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import Dataset
import numpy as np
from sklearn.metrics import classification_report

model_path = "./bert_model"
if not os.path.isdir(model_path):
    raise FileNotFoundError(f"Model directory '{model_path}' not found. Please train the model first.")

print(f"Loading model and tokenizer from {model_path}...")
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

df = pd.read_csv("my-track-a.csv")
df['text'] = df['text'].astype(str)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt")

dataset = Dataset.from_pandas(df[['text']])
dataset = dataset.map(tokenize, batched=True)

predictions = []
label_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']

with torch.no_grad():
    for i in range(len(dataset)):
        inputs = {k: torch.tensor(dataset[i][k]).unsqueeze(0) for k in ['input_ids', 'attention_mask']}
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().squeeze().tolist()
        predictions.append(preds)

pred_df = pd.DataFrame(predictions, columns=label_cols)
result_df = pd.concat([df['text'], pred_df], axis=1)
predictions_file = "predictions_bert.csv"
result_df.to_csv(predictions_file, index=False)
print(f"✅ Predictions saved to '{predictions_file}'.")

true_labels = df[label_cols].astype(int)
report = classification_report(true_labels, pred_df, target_names=label_cols, zero_division=0)

report_file = "bert_evaluation_report.txt"
with open(report_file, "w") as f:
    f.write(report)

print(f"✅ Evaluation report saved to '{report_file}'.")

