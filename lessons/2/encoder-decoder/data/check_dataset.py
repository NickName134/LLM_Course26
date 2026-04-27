from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0")
print(dataset["train"].column_names)
print(dataset["train"][0])  # vedi un esempio completo