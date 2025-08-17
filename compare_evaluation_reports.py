import os
import re
import pandas as pd
from tabulate import tabulate 

bert_report_file = "bert_evaluation_report.txt"
regression_report_file = "regression_model_classification_report.txt"

missing_files = []
if not os.path.isfile(bert_report_file):
    missing_files.append(bert_report_file)
if not os.path.isfile(regression_report_file):
    missing_files.append(regression_report_file)

if missing_files:
    print("⚠️ Warning: Missing evaluation report files:")
    for f in missing_files:
        print(f" - {f}")
    print("\nPlease run predictions and save evaluation reports before running this script.")
    exit(1)

def parse_classification_report(text):
    results = {}
    pattern = re.compile(r"^\s*(\S+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+\d+", re.MULTILINE)
    for match in pattern.finditer(text):
        label = match.group(1)
        precision = float(match.group(2))
        recall = float(match.group(3))
        f1 = float(match.group(4))
        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
        }
    return results

with open(bert_report_file, "r") as f:
    bert_text = f.read()

with open(regression_report_file, "r") as f:
    reg_text = f.read()

bert_metrics = parse_classification_report(bert_text)
reg_metrics = parse_classification_report(reg_text)

all_labels = sorted(set(bert_metrics.keys()) | set(reg_metrics.keys()))

rows = []
for label in all_labels:
    bert_vals = bert_metrics.get(label, {"precision": None, "recall": None, "f1-score": None})
    reg_vals = reg_metrics.get(label, {"precision": None, "recall": None, "f1-score": None})

    rows.append({
        "Emotion": label,
        "BERT Precision": bert_vals["precision"],
        "Regression Precision": reg_vals["precision"],
        "BERT Recall": bert_vals["recall"],
        "Regression Recall": reg_vals["recall"],
        "BERT F1-Score": bert_vals["f1-score"],
        "Regression F1-Score": reg_vals["f1-score"],
    })

df = pd.DataFrame(rows)

print("\nComparison of BERT vs Regression Model Performance:\n")
print(tabulate(df, headers='keys', tablefmt='grid', floatfmt=".2f"))

df.to_csv("comparison_report.csv", index=False)
print("\n✅ Comparison report saved as 'comparison_report.csv'")

print("\nComparison of BERT vs Regression Model Performance:\n")
print(tabulate(df, headers='keys', tablefmt='grid', floatfmt=".2f"))

df.to_csv("comparison_report.csv", index=False)
print("\n✅ Comparison report saved as 'comparison_report.csv'")

table_str = tabulate(df, headers='keys', tablefmt='grid', floatfmt=".2f")
with open("table_form_comparison_report.txt", "w") as f:
    f.write(table_str)

print("✅ Comparison report saved as 'comparison_report.txt'")