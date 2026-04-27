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
    tok, train_loader, val_loader = build_dataloader(max_seq_len=32, batch_size=4, shuffle=True)
    print(f"Vocabolario: {tok.vocab_size()} token")
    print(f"Batch per epoca: {len(train_loader)}\n")

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

        # training
        model.train()
        total_train_loss = 0.0
        train_correct    = 0
        train_total      = 0

        for input_ids, attention_mask, labels in train_loader:

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
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # ----------------------------------------------------------
            # Metriche — non modificare
            # ----------------------------------------------------------
            total_train_loss += loss.item()
            preds       = logits.argmax(dim=-1)
            train_correct    += (preds == labels).sum().item()
            train_total      += labels.size(0)
        #validazione
        model.eval()
        total_val_loss = 0.0
        val_correct    = 0      
        val_total      = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids      = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels         = labels.to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                # ----------------------------------------------------------
                # Metriche — non modificare
                # ----------------------------------------------------------

                total_val_loss += loss.item()
                preds       = logits.argmax(dim=-1)
                val_correct    += (preds == labels).sum().item()
                val_total      += labels.size(0)

        train_avg_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total * 100
        val_avg_loss   = total_val_loss / len(val_loader)
        val_accuracy   = val_correct / val_total * 100

        print(f"Epoca {epoch+1:>3}/{num_epochs} | "
            f"train loss {train_avg_loss:.4f} acc {train_accuracy:.1f}% | "
            f"val loss {val_avg_loss:.4f} acc {val_accuracy:.1f}%")
    print("\nTraining completato.")

    torch.save(model.state_dict(), "checkpoints/encoder_classifier.pt")
    print("Modello salvato -> checkpoints/encoder_classifier.pt")

    

if __name__ == "__main__":
    train()