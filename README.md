# Fault Detection with Shared-LSTM (Option B)

Implementation of the extra-task for **Fundamentos de Redes Neurais** – UFMA.
The objective is to detect failures in multi-sensor vibration data.

## Project structure

```
├── codes/              # Core Python modules
│   ├── data_utils.py   # Loading + window generation
│   ├── LSTM.py         # Model definition / training helpers
│   ├── train_lstm.py   # End-to-end training script
│   └── ...
├── dataset/            # CSV parts provided by professor
├── experiments/        # Outputs (models, plots, metrics)
├── requirements.txt    # Pip dependencies
└── environment.yml     # Conda equivalent
```

## Quick start (pip)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m codes.train_lstm \
    --dataset_dir dataset \
    --window 240 --step 1 \
    --batch 128 --epochs 100 \
    --exp_dir experiments/exp01
```

### Optional flags
* `--focal` – use `BinaryFocalCrossentropy` loss (better for class-imbalance).
* `--debug` – verbose logs.

## Conda environment
If you prefer conda / mamba:

```bash
mamba env create -f environment.yml
conda activate fault-lstm
```

## Notebook
Open `notebooks/eda_train_report.ipynb` (to be generated) for:
1. Exploratory Data Analysis (EDA)
2. Window creation rationale
3. Model training with early stopping & LR scheduler
4. Confusion matrix, ROC & PR curves using `ModelEvaluator`
5. Discussion of limitations & future work

---
© 2025 Thales Valente – Teaching purposes
