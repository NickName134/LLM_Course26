# model.py
import math
import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Positional Encoding
# ------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Aggiunge informazione di posizione agli embedding.

    Usa funzioni seno/coseno a frequenze diverse: ogni posizione
    ha un pattern unico che il modello può imparare a leggere.
    Non ha parametri allenabili: è un calcolo deterministico.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # matrice [max_len, d_model] — una riga per posizione
        pe = torch.zeros(max_len, d_model)

        # vettore colonna [max_len, 1] con gli indici di posizione
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # fattori di scala per le frequenze
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # dimensioni pari -> seno, dimensioni dispari -> coseno
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # aggiunge dimensione batch: [1, max_len, d_model]
        # register_buffer: non è un parametro allenabile ma viene
        # spostato su GPU insieme al modello con .to(device)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, d_model]
        # pe shape: [1, max_len, d_model]
        # somma solo le prime seq_len posizioni
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ------------------------------------------------------------------
# Encoder-only Transformer
# ------------------------------------------------------------------

class EncoderClassifier(nn.Module):
    """
    Transformer encoder-only per classificazione binaria.

    Architettura:
      Embedding -> PositionalEncoding -> TransformerEncoder -> CLS -> Linear
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()

        # TODO 1 ──────────────────────────────────────────────────
        # Definisci i quattro componenti del modello in __init__:
        #
        #   self.embedding   -> nn.Embedding(vocab_size, d_model)
        #   self.pos_encoder -> PositionalEncoding(d_model, dropout)
        #   self.encoder     ->  composto da:
        #                        encoder_layer = nn.TransformerEncoderLayer(
        #                            d_model=d_model,
        #                            nhead=nhead,
        #                            dim_feedforward=dim_feedforward,
        #                            dropout=dropout,
        #                            batch_first=True,   <- importante
        #                            norm_first=True,
        #                            activation="gelu"
        #                        )
        #                        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        #   self.classifier  -> nn.Linear(d_model, num_classes)
        #
        # Nota: batch_first=True significa che il tensore ha shape
        # [batch, seq_len, d_model] — più intuitivo di [seq_len, batch, d_model]
        self.embedding = nn.Embedding(vocab_size, d_model) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, # dimensione degli embedding
                nhead=nhead, # numero di teste di attenzione
                dim_feedforward=dim_feedforward, # dimensione del feedforward interno
                dropout=dropout, 
                batch_first=True, # importante per avere shape [batch, seq_len, d_model]
                norm_first=True,
                activation="gelu"
            )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,        # [batch, seq_len]
        attention_mask: torch.Tensor,   # [batch, seq_len]  1=reale 0=pad
    ) -> torch.Tensor:                  # [batch, num_classes]

        # TODO 2 ──────────────────────────────────────────────────
        # Scrivi il forward pass in 4 passi:
        #
        #   1. embedding: passa input_ids attraverso self.embedding
        #      risultato shape: [batch, seq_len, d_model]
        #
        #   2. positional encoding: passa il risultato a self.pos_encoder
        #      risultato shape: [batch, seq_len, d_model]  (invariato)
        #
        #   3. encoder: passa a self.encoder con la src_key_padding_mask.
        #      ATTENZIONE: TransformerEncoder si aspetta True dove ignorare,
        #      quindi la maschera va invertita:
        #        src_key_padding_mask = (attention_mask == 0)
        #      risultato shape: [batch, seq_len, d_model]
        #
        #   4. CLS token: prendi solo il primo token
        #        cls_output = out[:, 0, :]
        #      risultato shape: [batch, d_model]
        #
        #   5. classificatore: passa cls_output a self.classifier
        #      risultato shape: [batch, num_classes]
        #
        # Restituisci i logit finali.
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0)) # invertiamo la maschera per il padding
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


# ------------------------------------------------------------------
# Test rapido
# ------------------------------------------------------------------

if __name__ == "__main__":
    # parametri piccoli per girare veloce su CPU
    model = EncoderClassifier(
        vocab_size=100,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=128,
        num_classes=2,
    )

    # conta i parametri allenabili
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri allenabili: {n_params:,}")

    # forward pass con dati fittizi
    batch_size, seq_len = 4, 16
    input_ids      = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    logits = model(input_ids, attention_mask)
    print(f"Input shape:  {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output (logits):\n{logits}")