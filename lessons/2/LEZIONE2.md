# STEP 1
Creare un progetto con la seguente struttura:

```
myproject/
├── config.py         TASK = "summarization"/"translation"/"simplification"
├── data/
│   └── dataset.py
│  
├── model/
│   └── model.py      uguale per tutti i task
├── training/
│   ├── metrics.py    metriche diverse per task diversi
│   └── trainer.py    uguale per tutti i task
├── train.py          legge TASK da config e carica il modulo giusto
└── predict.py

```

