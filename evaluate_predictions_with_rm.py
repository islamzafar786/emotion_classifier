import pandas as pd
from sklearn.metrics import classification_report

true_df = pd.read_csv("my-track-a.csv")
pred_df = pd.read_csv("predictions.csv")

label_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']

true_labels = true_df[label_cols].astype(int)
pred_labels = pred_df[label_cols].astype(int)

report = classification_report(true_labels, pred_labels, target_names=label_cols, zero_division=0)

print("Classification Report:\n")
print(report)

with open("regression_model_classification_report.txt", "w") as f:
    f.write("Classification Report:\n\n")
    f.write(report)
