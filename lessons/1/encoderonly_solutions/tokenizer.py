# tokenizer.py
import re
from collections import Counter
from typing import List

class SimpleTokenizer:
    """
    Tokenizer minimale a parole intere.
    Non è sofisticato: non gestisce le parole composte, i prefissi/suffissi,
    o le parole fuori vocabolario in modo intelligente, ma è semplice e trasparente
    """

    # Token speciali: indici fissi, sempre gli stessi
    PAD = "[PAD]"   # padding: riempie le sequenze corte
    UNK = "[UNK]"   # unknown: parole fuori vocabolario
    CLS = "[CLS]"   # classification token: va sempre all'inizio

    def __init__(self):
        self.vocab = {}       # parola  -> indice
        self.inv_vocab = {}   # indice  -> parola

    # ------------------------------------------------------------------
    # 1. Pre-processing del testo grezzo
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """
        Trasforma una stringa in una lista di token (parole).

        Passi:
          1. lowercase
          2. separa la punteggiatura dal testo con uno spazio
             (così "bello!" diventa ["bello", "!"] e non ["bello!"])
          3. split sugli spazi
          4. rimuove token vuoti

        Nota: gli apostrofi italiani vengono gestiti separando
        la parola in due token: "l'uomo" -> ["l'", "uomo"].
        """

        text = text.lower()
        #lowercase per normalizzare

        # separa punteggiatura con spazi
        text = re.sub(r"([,.!?;:()\"\-])", r" \1 ", text)
        # split e pulizia
        # re.sub(pattern, sostituzione, stringa) — sostituisce tutte le occorrenze
        # del pattern con la sostituzione nella stringa.
        #
        # pattern: r"([,.!?;:()\"\-])"
        #   - le parentesi tonde () creano un "gruppo di cattura"
        #   - dentro ci sono tutti i caratteri di punteggiatura che vogliamo separare
        #   - r"..." è una raw string — i backslash non vengono interpretati come escape
        #
        # sostituzione: r" \1 "
        #   - \1 si riferisce al primo gruppo di cattura — cioè il carattere trovato
        #   - lo circonda con spazi: "bello!" -> "bello ! "
        #
        # Perché separare la punteggiatura invece di rimuoverla?
        # Perché "!" e "?" portano informazione sul tono della frase.
        # Separandola diventa un token autonomo invece di sparire.
        tokens = [t for t in text.split() if t]
        # text.split() divide la stringa su ogni spazio e restituisce una lista
        return tokens

    # ------------------------------------------------------------------
    # 2. Costruzione del vocabolario
    # ------------------------------------------------------------------

    def build_vocab(self, texts: List[str], max_vocab: int = 10000) -> None:
        """
        Costruisce il vocabolario a partire da una lista di frasi.

        Ordine degli indici:
          0 → [PAD]
          1 → [UNK]
          2 → [CLS]
          3, 4, 5, ... → parole ordinate per frequenza (dalla più comune)

        Perché i token speciali vengono prima?
        Convenzione universale — indici bassi e fissi li rendono
        facili da riconoscere e da escludere dal calcolo della loss.
        """
        # TODO 1 ──────────────────────────────────────────────────────
        # Conta la frequenza di ogni token nel corpus.
        # Suggerimento: usa Counter e il metodo _tokenize su ogni frase.
        # Poi tieni solo i max_vocab - 3 token più frequenti
        # (sottrai 3 per fare spazio ai token speciali).
        #
        # Alla fine costruisci self.vocab e self.inv_vocab.
        # Ricorda: i token speciali vanno agli indici 0, 1, 2.
        #
        # Input:  texts = ["il film è bello", "non mi è piaciuto", ...]
        # Output: self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2,
        #                       "il": 3, "film": 4, ...
        token_freq = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            token_freq.update(tokens)

        most_common = token_freq.most_common(max_vocab - 3)

        self.vocab = {self.PAD: 0, self.UNK: 1, self.CLS: 2}
        for i, (token, _) in enumerate(most_common, start=3):
            self.vocab[token] = i

        self.inv_vocab = {v: k for k, v in self.vocab.items()}




    # ------------------------------------------------------------------
    # 3. Encode e decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """
        Testo -> lista di interi.
        Le parole fuori vocabolario diventano l'indice di [UNK].

        Nota: NON aggiunge [CLS] qui, lo fa data.py,
        così la responsabilità è chiara: il tokenizer traduce,
        data.py prepara le sequenze per il modello.
        """
        # TODO 2 ──────────────────────────────────────────────────────
        # Tokenizza il testo con _tokenize, poi mappa ogni token
        # al suo indice in self.vocab.
        # Se il token non è nel vocabolario, usa l'indice di [UNK].
        #
        # Input:  "il film è bello"
        # Output: [3, 4, 1, 5]   (esempio — gli indici dipendono dal vocab)
        tokens = self._tokenize(text)
        ids = [self.vocab.get(token, self.vocab[self.UNK]) for token in tokens]
        return ids
        

    def decode(self, ids: List[int]) -> str:
        """
        Lista di interi -> testo.
        Usato principalmente per debug: permette di leggere
        cosa sta "vedendo" il modello dopo tokenizzazione e padding.
        """
        tokens = [self.inv_vocab.get(i, self.UNK) for i in ids]
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # 4. Proprietà utili
    # ------------------------------------------------------------------

    def vocab_size(self) -> int:
        return len(self.vocab)

    def pad_id(self) -> int:
        return self.vocab[self.PAD]

    def cls_id(self) -> int:
        return self.vocab[self.CLS]


# ------------------------------------------------------------------
# Test rapido — esegui questo file direttamente per verificare
# ------------------------------------------------------------------
if __name__ == "__main__":
    corpus = [
        "il film è bellissimo davvero",
        "storia noiosa e recitazione pessima",
        "mi ha emozionato molto",
        "non lo consiglio a nessuno",
        "capolavoro assoluto del cinema italiano",
        "una perdita di tempo totale",
        "attori bravissimi e regia curata",
        "trama confusa e finale deludente",
    ]

    tok = SimpleTokenizer()
    tok.build_vocab(corpus)

    print(f"Vocabolario: {tok.vocab_size()} token")
    print(f"Prime 10 voci: {list(tok.vocab.items())[:10]}")
    print()

    frase = "il film è bellissimo"
    ids = tok.encode(frase)
    print(f"encode('{frase}') -> {ids}")
    print(f"decode({ids})     -> '{tok.decode(ids)}'")
    print()

    # test parola sconosciuta
    ids_unk = tok.encode("questo film è fantasmagorico")
    print(f"encode con parola sconosciuta -> {ids_unk}")
    print(f"decode -> '{tok.decode(ids_unk)}'")