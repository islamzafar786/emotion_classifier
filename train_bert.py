import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import transformers

print("🚀 Starting BERT training script...")
print("Transformers version:", transformers.__version__)

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
else:
    print("Warning: No GPU detected, training will be slower.")

df = pd.read_csv("my-track-a.csv")
df['text'] = df['text'].astype(str)
print(f"✅ Loaded dataset with {len(df)} samples.")

label_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']

mlb = MultiLabelBinarizer()
df['labels'] = df[label_cols].values.tolist()
df['labels'] = df['labels'].apply(lambda x: [float(i) for i in x])


print("Loading tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_texts, val_texts = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}")

train_ds = Dataset.from_pandas(train_texts[['text', 'labels']])
val_ds = Dataset.from_pandas(val_texts[['text', 'labels']])

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print("Datasets tokenized and ready.")

print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)

training_args = TrainingArguments(
    output_dir="./bert-output",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none", 
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    precision = (preds & labels).sum() / np.maximum(preds.sum(), 1)
    recall = (preds & labels).sum() / np.maximum(labels.sum(), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting training...")

trainer.train()

print("Training finished.")

# Save model and tokenizer
model.save_pretrained("./bert_model")
tokenizer.save_pretrained("./bert_model")
print("✅ BERT model and tokenizer saved in ./bert_model")
