# train.py
import torch
import torch.nn as nn
from torch import optim
from data import build_dataloader
from model import EncoderClassifier


def train():

    # ------------------------------------------------------------------
    # 1. Setup dispositivo
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Dispositivo: {device}")

    # ------------------------------------------------------------------
    # 2. Dati
    # ------------------------------------------------------------------
    tok, loader = build_dataloader(max_seq_len=32, batch_size=4, shuffle=True)
    print(f"Vocabolario: {tok.vocab_size()} token")
    print(f"Batch per epoca: {len(loader)}\n")

    # ------------------------------------------------------------------
    # 3. Modello
    # ------------------------------------------------------------------
    model = EncoderClassifier(
        vocab_size=tok.vocab_size(),
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri allenabili: {n_params:,}\n")

    # ------------------------------------------------------------------
    # 4. Loss e optimizer
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    num_epochs = 20

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for input_ids, attention_mask, labels in loader:

            # sposta i tensori sul dispositivo corretto
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)

            # TODO ──────────────────────────────────────────────────
            # Scrivi i 5 passi del training loop:
            #
            #   1. azzera i gradienti
            #      optimizer.zero_grad()
            #
            #   2. forward pass
            #      logits = model(input_ids, attention_mask)
            #
            #   3. calcola la loss
            #      loss = criterion(logits, labels)
            #
            #   4. backward pass — calcola i gradienti
            #      loss.backward()
            #
            #   5. aggiorna i pesi
            #      optimizer.step()
            pass

            # ----------------------------------------------------------
            # Metriche — non modificare
            # ----------------------------------------------------------
            total_loss += loss.item()
            preds       = logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total * 100
        print(f"Epoca {epoch+1:>3}/{num_epochs} | loss {avg_loss:.4f} | accuracy {accuracy:.1f}%")

    print("\nTraining completato.")
    torch.save(model.state_dict(), "checkpoints/encoder_classifier.pt")
    print("Modello salvato -> checkpoints/encoder_classifier.pt")


if __name__ == "__main__":
    train()