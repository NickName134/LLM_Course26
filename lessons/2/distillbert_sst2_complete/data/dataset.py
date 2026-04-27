# data/dataset.py

from datasets import load_dataset
from transformers import AutoTokenizer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__))) # aggiungiamo la cartella padre al path
from config import(
    MODEL,
    DATASET,
    DATASET_CONFIG,
    MAX_LENGTH
)

def load_data():
    # Carichiamo il dataset usando Hugging Face Datasets
    dataset = load_dataset(DATASET, DATASET_CONFIG)
    print(dataset)  # stampiamo le informazioni sul dataset per verifica
    # Carichiamo il tokenizer pre-addestrato di DistilBERT
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Tokenizziamo il dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],  # la colonna con i testi si chiama 'sentence' in SST-2
            padding='max_length',  # aggiungiamo padding fino a MAX_LENGTH
            truncation=True,       # trunchiamo se la sequenza è più lunga di MAX_LENGTH
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True, 
        remove_columns=['sentence','idx'])  # rimuoviamo la colonna originale dei testi
    
    tokenized_dataset["train"]      = tokenized_dataset["train"].select(range(1000))
    tokenized_dataset["validation"] = tokenized_dataset["validation"].select(range(100))

    return tokenized_dataset, tokenizer

if __name__ == "__main__":
    tokenized_dataset, tokenizer = load_data()
    print(tokenized_dataset["train"][:5])  # stampiamo i primi 5 esempi tokenizzati per verifica
    print(tokenizer)  # stampiamo il tokenizer per verifica