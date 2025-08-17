# Emotion Classifier (dataset – Track A) Dataset Download instructions are defined below.

Multi-label emotion detection for text (**anger, fear, joy, sadness, surprise, disgust**).  
This project implements and compares two approaches:
- **Regression baseline** (TF-IDF + Logistic Regression)  
- **BERT-based model** (fine-tuned transformer)  

Project for *Introduction to Natural Language Processing*.

---

## Project Structure
project-emotion_classifier/
├─ __pycache__/                  # Python cache files (ignored)
├─ .venv/                        # Virtual environment (ignored)
├─ bert_model/                   # Saved fine-tuned BERT weights (ignored)
├─ bert-output/                  # BERT training logs/outputs (ignored)
│
├─ main.py                       # Entry point with predict() function
├─ train.py                      # Train Regression model
├─ train_bert.py                 # Train BERT model
├─ predict_with_bert.py          # Predict labels using BERT
├─ evaluate_predictions_with_rm.py  # Evaluate Regression model predictions
├─ compare_evaluation_reports.py # Compare Regression vs BERT results
│
├─ model.pkl                     # Saved regression model (ignored in repo)
├─ vectorizer.pkl                 # Saved TF-IDF vectorizer (ignored in repo)
│
├─ requirements.txt              # Python dependencies
├─ regression_model_classification_report.txt   # Regression evaluation results
├─ bert_evaluation_report.txt    # BERT evaluation results
├─ comparison_report.csv         # Side-by-side comparison of models
├─ table_form_comparison_report.txt # Alternative comparison in table form
│
├─ predictions.csv               # Regression predictions
├─ predictions_bert.csv          # BERT predictions
│
├─ data/
│   ├─ my-track-a.csv            # Training dataset (do not commit full dataset)
│
├─ test_training_args.py         # Training configs for testing
├─ test_init_args.py             # Init configs for testing
└─ ...


---

## ⚙️ Setup

### 1) Create & activate virtual environment

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate




### 2) Install dependencies
        python -m pip install --upgrade pip setuptools wheel
        pip install -r requirements.txt


If requirements.txt is not available, install manually:
python -m pip install \
    transformers==4.53.0 \
    datasets \
    torch==2.7.1 \
    scikit-learn==1.7.0 \
    accelerate==1.8.1 \
    joblib \
    pandas \
    numpy \
    regex \
    tqdm \
    huggingface-hub \
    tokenizers==0.21.2 \
    safetensors \
    packaging \
    requests \
    tabulate


## 📥 Download Required Files

Some files including used dataset are too large to store directly in this repository.  
Please download them from Google Drive and place them inside the project.  

👉 [Download all required files (Google Drive link)](https://drive.google.com/file/d/1lsobpCX-h3p_Ci0jrsVwxC9I9ot3A2zG/view?usp=drive_link)

After downloading, extract the archive and place the contents


Finally, run all commands inside your activated virtual environment.

1) If you want to train the model yourself. Run this command to train BERT.
    *** python train_bert.py ***   # Saves fine-tuned weights + evaluation report.
2) To Predict with BERT
    *** python predict_with_bert.py ***   # Generates predictions (predictions_bert.csv).
3) To evaluate Regression model predictions
    *** python evaluate_predictions_with_rm.py ***   # Computes metrics for regression model (regression_model_classification_report.txt).
4) To compare Reports (BERT vs Regression)
    *** python compare_evaluation_reports.py ***   # Outputs comparison (comparison_report.csv).
  


