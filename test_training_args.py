from transformers import TrainingArguments
print(TrainingArguments.__init__.__code__.co_varnames)

training_args = TrainingArguments(
    output_dir="./test-output",
    evaluation_strategy="epoch"
)

print("TrainingArguments imported and instantiated successfully.")
