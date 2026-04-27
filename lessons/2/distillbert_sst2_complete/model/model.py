# model/model.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import MODEL, NUM_LABELS

from transformers import AutoModelForSequenceClassification

def load_model():
    # Carichiamo il modello pre-addestrato per la classificazione
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=NUM_LABELS
    )

    print(f"Modello caricato: {MODEL}")
    print(f"Parametri totali: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Parametri allenabili: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model