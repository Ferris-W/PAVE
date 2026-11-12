# PAVE: Perturb-and-Preserve

Official implementation of **“Perturb-and-preserve: Learning Transferable Spatial Dependencies for Inductive Spatio-Temporal Forecasting.”**

## Setup

Run all commands from this directory:

```bash
python -m pip install -r requirements.txt
```

## Data

Download the prepared datasets from [Google Drive](https://drive.google.com/drive/folders/1qoMflWKEt2QMNrBNb_7mxVoD1ybaBoZ_?usp=drive_link) and place the four dataset folders in `data/` as shown below.

```text
data/
├── PEMS03/{PEMS03.npz,PEMS03.csv,adj_PEMS03.pkl}
├── PEMS07/{PEMS07.npz,PEMS07.csv,adj_PEMS07.pkl}
├── PEMS08/{PEMS08.npz,PEMS08.csv,adj_PEMS08.pkl}
└── CHENGDU/{CHENGDU.npz,CHENGDU.csv,adj_CHENGDU.pkl}
```

## Evaluate

Evaluate the saved teacher and student:

```bash
python scripts/experiment.py evaluate --dataset [DATASET] --device cuda:0
```

## Train

Train the teacher alone:

```bash
python scripts/experiment.py train --dataset [DATASET] --role teacher --device cuda:0
```

Train the PAVE student using the saved teacher:

```bash
python scripts/experiment.py train --dataset [DATASET] --role pave --teacher-source saved --device cuda:0
```

Train teacher and student:

```bash
python scripts/experiment.py train --dataset [DATASET] --teacher-source fresh --device cuda:0
```

Replace `[DATASET]` with the dataset name.

