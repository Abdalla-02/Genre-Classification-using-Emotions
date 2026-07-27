# Python-Umgebung — Setup-Anleitung

## Warum eine isolierte Umgebung?

Ohne virtuelle Umgebung installierst du Pakete global — das führt schnell zu Versionskonflikten
zwischen Projekten und macht deine Ergebnisse für andere (und dich selbst in drei Monaten)
schwer reproduzierbar. Für eine Bachelorarbeit ist eine sauber dokumentierte Umgebung Teil der
wissenschaftlichen Nachvollziehbarkeit.

## Option A: venv (einfach, kein extra Tool nötig)

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

## Option B: conda (empfohlen, wenn du bereits Anaconda/Miniconda nutzt)

```bash
conda env create -f environment.yml
conda activate ba-genre-emotion
```

## Wichtiger Hinweis: PyTorch vs. TensorFlow

AST, MusiCNN und torchopenl3 laufen alle auf **PyTorch** — bewusst so gewählt, damit du nur
ein Deep-Learning-Framework brauchst. Falls du YAMNet (TensorFlow-basiert) unbedingt als
zusätzliche Baseline testen willst, empfehle ich eine **zweite, separate Umgebung**:

```bash
python3.11 -m venv .venv-tf
source .venv-tf/bin/activate
pip install tensorflow==2.16.1 tensorflow-hub==0.16.1 numpy pandas
```

Mische niemals torch und tensorflow in derselben Umgebung ohne Not — die CUDA-Versionsanforderungen
kollidieren häufig und die Fehlersuche kostet unnötig Zeit, die du für die Experimente brauchst.

## Dependency-Management: Best Practices für die Thesis

1. **Versionen immer fest pinnen** (`==`, nicht `>=`) — sonst installiert sich in drei Monaten
   eine andere Version und deine Ergebnisse sind evtl. nicht mehr exakt reproduzierbar.
2. **requirements.txt / environment.yml ins Git-Repository committen** — das ist Teil deiner
   methodischen Nachvollziehbarkeit (kannst du auch im Anhang der Thesis erwähnen).
3. **Nach jeder größeren Änderung aktualisieren:**
   ```bash
   pip freeze > requirements_freeze.txt   # exakter Snapshot aller installierten Pakete
   ```
   Das unterscheidet sich von `requirements.txt` (deine bewusst gewählten Top-Level-Pakete) —
   `requirements_freeze.txt` enthält zusätzlich alle transitiven Abhängigkeiten. Für die Thesis
   reicht meist die kuratierte `requirements.txt`.
4. **Random Seeds fixieren** für Reproduzierbarkeit (gehört strenggenommen nicht zum
   Dependency-Management, ist aber genauso wichtig):
   ```python
   import random, numpy as np, torch
   SEED = 42
   random.seed(SEED)
   np.random.seed(SEED)
   torch.manual_seed(SEED)
   ```

## Projektstruktur-Empfehlung

```
thesis-project/
├── environment.yml / requirements.txt
├── README_ENV.md
├── data/
│   ├── raw/              # Original-CSVs, nie verändern
│   └── processed/        # bereinigte/erweiterte Daten
├── src/
│   ├── features/         # Feature-Extraktion (AST, OpenL3, ...)
│   ├── models/            # Emotion-Regression, Genre-Klassifikation
│   └── evaluation/        # Metriken, GroupKFold, Plots
├── notebooks/              # explorative Analyse (Cohen's d, Visualisierungen)
├── results/                # Modell-Outputs, Metriken, Plots
└── thesis/                 # LaTeX/Word-Dateien der schriftlichen Arbeit
```

Diese Struktur trennt sauber zwischen Rohdaten, Code und Ergebnissen — hilfreich, wenn der
Betreuer nach Reproduzierbarkeit fragt.
