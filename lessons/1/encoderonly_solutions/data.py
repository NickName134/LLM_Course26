# data.py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from tokenizer import SimpleTokenizer


# ------------------------------------------------------------------
# Dataset italiano — recensioni di film
# etichetta 1 = positivo, 0 = negativo
# ------------------------------------------------------------------

def get_corpus():
    train = [
        ("il film è bellissimo davvero",            1),
        ("storia noiosa e recitazione pessima",      0),
        ("mi ha emozionato moltissimo",              1),
        ("non lo consiglio a nessuno",               0),
        ("capolavoro assoluto del cinema italiano",  1),
        ("una perdita di tempo totale",              0),
        ("attori bravissimi e regia curata",         1),
        ("trama confusa e finale deludente",         0),
        ("dialoghi brillanti e storia coinvolgente", 1),
        ("fotografia splendida e musiche bellissime",1),
        ("personaggi piatti e sceneggiatura debole", 0),
        ("ritmo lento e storia prevedibile",         0),
    ]
    val = [
        ("finale emozionante e inaspettato",          1),
        ("effetti speciali mediocri e trama assente", 0),
        ("regia magistrale e interpretazioni intense",1),
        ("film inutile e privo di senso",             0),
    ]
    return train, val


# ------------------------------------------------------------------
# Dataset PyTorch
# ------------------------------------------------------------------

class ReviewDataset(Dataset): # definiamo una classe che estende Dataset di PyTorch
    """
    Prende il corpus e lo trasforma in tensori
    pronti per il modello.

    Ogni campione è una tupla:
      - input_ids:      LongTensor [max_seq_len]
      - attention_mask: LongTensor [max_seq_len]
      - label:          LongTensor scalare (0 o 1)
    """

    def __init__(self, tokenizer: SimpleTokenizer, split: str = "train", max_seq_len: int = 32):
        super().__init__()
        self.tokenizer = tokenizer # usiamo il tokenizer costruito in tokenizer.py
        self.max_seq_len = max_seq_len # lunghezza massima della sequenza (inclusi padding e [CLS])
        train, val = get_corpus() # carichiamo il corpus, che è una lista di tuple (testo, etichetta)
        self.data =  train if split == "train" else val # carichiamo il corpus

    # questi due metodi vengono chiamati da DataLoader per iterare sul dataset (non da noi direttamente)
    def __len__(self) -> int: # restituisce il numero di campioni nel dataset
        return len(self.data)

    def __getitem__(self, idx: int): # restituisce il campione alla posizione idx
        text, label = self.data[idx]

        # TODO 1 ──────────────────────────────────────────────────
        # Prepara la sequenza di input per il modello.
        #
        # Passi:
        #   1. encode del testo con self.tokenizer.encode(text)
        #   2. aggiungi [CLS] all'inizio:
        #      ids = [self.tokenizer.cls_id()] + ids
        #   3. tronca se troppo lunga:
        #      ids = ids[:self.max_seq_len]
        #   4. costruisci attention_mask: lista di 1, stessa lunghezza di ids
        #   5. fai padding di ids e attention_mask fino a max_seq_len
        #      con self.tokenizer.pad_id() per ids e 0 per la maschera
        #
        # Alla fine ids e attention_mask devono avere
        # entrambi lunghezza esattamente max_seq_len.
        #
        # Input:  "il film è bello"  ->  ids grezzo: [4, 23, 7, 156]
        # Output: ids = [2, 4, 23, 7, 156, 0, 0, ...]  (lunghezza 32)
        #         mask= [1, 1,  1, 1,   1, 0, 0, ...]  (lunghezza 32)
        ids = self.tokenizer.encode(text) # restituisce una lista di interi
        ids = [self.tokenizer.cls_id()] + ids # aggiungiamo l'id di [CLS] all'inizio
        ids = ids[:self.max_seq_len] # tronchiamo se troppo lunga (ricorda: lista[start:end] include start ma esclude end)
        attention_mask = [1] * len(ids) # costruiamo la maschera di attenzione
        # facciamo padding fino a max_seq_len
        pad_length = self.max_seq_len - len(ids) # differenza tra lunghezza desiderata e lunghezza attuale       
        ids += [self.tokenizer.pad_id()] * pad_length
        attention_mask += [0] * pad_length
        

        # TODO 2 ──────────────────────────────────────────────────
        # Converti ids, attention_mask e label in tensori PyTorch.
        # Usa torch.tensor(..., dtype=torch.long) per tutti e tre.
        # Restituisci la tupla (input_ids, attention_mask, label).
        input_ids = torch.tensor(ids, dtype=torch.long) 
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return input_ids, attention_mask, label_tensor


# ------------------------------------------------------------------
# Funzione di utilità per creare tokenizer + dataloader in un colpo
# ------------------------------------------------------------------

def build_dataloader(
    max_seq_len: int = 32, 
    batch_size: int = 4, #
    shuffle: bool = True
) -> Tuple[SimpleTokenizer, DataLoader]:
    """
    Costruisce tokenizer, dataset e dataloader pronti per il training.
    Restituisce entrambi perché train.py ha bisogno del tokenizer
    per sapere il vocab_size da passare al modello.
    """
    train_data, _ = get_corpus()
    texts = [text for text, _ in train_data]

    tok = SimpleTokenizer()
    tok.build_vocab(texts)

    train_dataset = ReviewDataset(tok, split="train", max_seq_len=max_seq_len)
    val_dataset = ReviewDataset(tok, split="val", max_seq_len=max_seq_len)
    
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return tok, train_loader, val_loader


# ------------------------------------------------------------------
# Test rapido
# ------------------------------------------------------------------

if __name__ == "__main__":
    tok, train_loader, val_loader = build_dataloader(max_seq_len=16, batch_size=4)

    print(f"Vocabolario: {tok.vocab_size()} token")
    print(f"Dataset: {len(train_loader.dataset)} campioni")
    print(f"Batch: {len(train_loader)} batch da 4\n")

    # mostra il primo batch
    input_ids, attention_mask, labels = next(iter(train_loader))

    print(f"input_ids shape:      {input_ids.shape}")
    print(f"attention_mask shape: {attention_mask.shape}")
    print(f"labels shape:         {labels.shape}")
    print()

    # decodifica la prima frase del batch per verifica visiva
    prima_frase = input_ids[0].tolist()
    print(f"Prima frase (ids):    {prima_frase}")
    print(f"Prima frase (testo):  {tok.decode(prima_frase)}")
    print(f"Etichetta:            {labels[0].item()}")