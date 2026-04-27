# train.py

# Entry point del progetto: avvia il training completo.
# Esegui con: python train.py

import os
import sys
from config import OUTPUT_DIR, SEED
from data.dataset import load_data
from model.model import load_model
from training.metrics import compute_metrics
from training.trainer import build_trainer

import torch
import random
import numpy as np


def set_seed(seed: int):
    """
    Fissa il seed per la riproducibilità.
    Va impostato su tutte le librerie che usano numeri casuali.
    Senza questo ogni run produce risultati leggermente diversi.
    """
    random.seed(seed)
    # seed per la libreria standard Python — usata in alcune operazioni
    # di shuffling e campionamento

    np.random.seed(seed)
    # seed per numpy — usato nelle metriche e nel preprocessing

    torch.manual_seed(seed)
    # seed per PyTorch — usato nell'inizializzazione dei pesi
    # e nel dropout

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # seed per tutte le GPU — necessario se usi più GPU


def main():

    # -- 0. riproducibilità ---------------------------------------
    set_seed(SEED)
    print(f"Seed: {SEED}")

    # -- 1. dati --------------------------------------------------
    print("\n--- Caricamento dataset ---")
    tokenized_dataset, tokenizer = load_data()
    print(f"Train:      {len(tokenized_dataset['train'])} esempi")
    print(f"Validation: {len(tokenized_dataset['validation'])} esempi")

    # -- 2. modello -----------------------------------------------
    print("\n--- Caricamento modello ---")
    model = load_model()

    # -- 3. trainer -----------------------------------------------
    print("\n--- Configurazione trainer ---")
    trainer = build_trainer(
        model=model,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -- 4. training ----------------------------------------------
    print("\n--- Avvio training ---")
    trainer.train()
    # il Trainer stampa automaticamente:
    # - la loss ad ogni LOGGING_STEPS step
    # - le metriche di valutazione alla fine di ogni epoca
    # - il tempo stimato rimanente

    # -- 5. valutazione finale ------------------------------------
    print("\n--- Valutazione finale ---")
    risultati = trainer.evaluate()
    print(f"Accuracy finale: {risultati['eval_accuracy']:.4f}")

    # -- 6. salva il modello finale -------------------------------
    print(f"\n--- Salvataggio modello ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    # salviamo anche il tokenizer insieme al modello —
    # serve per caricare il modello in predict.py senza
    # dover sapere quale tokenizer era stato usato
    print(f"Modello salvato in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
