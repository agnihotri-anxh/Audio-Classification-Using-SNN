# Audio Classification Using Spiking Neural Networks (SNN)

This project trains a Spiking Neural Network (SNN) to classify environmental sounds from the UrbanSound8K dataset using MFCC features encoded into spike trains via population threshold encoding.

## Environment
- Python (see `requirements.txt`)
- PyTorch, snntorch, librosa, scikit-learn, matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data
Place UrbanSound8K audio and metadata under `Data/` as used in the notebook. Large audio folds and models are ignored via `.gitignore`.

## Training
All code is in `Claasification.ipynb`. Key hyperparameters:
- Epochs: 100 (early stopping patience 10)
- Batch size: 32
- MFCCs: 40; thresholds: 15 → input size 600
- Hidden units: 256; dropout 0.6
- Optimizer: Adam (1e-4)

Run the notebook cells to extract features, encode spikes, train, and evaluate.

## Results
- Final Test Accuracy: **63.19%**
- Macro ROC-AUC: **0.817**
- Micro ROC-AUC: **0.837**

The best model checkpoint is saved as `best_snn_model.pth` during training.

## Curves
Training/validation curves are produced by the notebook:
- Accuracy vs Epochs
- Loss vs Epochs

If you save figures during training, place them under `assets/` and reference here:
- `assets/accuracy_curve`
- `assets/loss_curve`
- <img width="996" height="470" alt="image" src="https://github.com/user-attachments/assets/b6f1d2b4-f89c-4a9a-99d2-9790efae11dd" />


## Confusion Matrix & ROC-AUC
The notebook also renders a confusion matrix and multiclass ROC curves for per-class AUC and micro-average AUC.

## Reproduce
1. Ensure `Data/UrbanSound8K.csv` and folds are available locally.
2. Run `Claasification.ipynb` end-to-end.
3. Evaluate with the provided test loop to verify accuracy and curves.

## Notes
- We normalize MFCCs to [0,1] and encode with 15 thresholds to generate spike trains while preserving temporal dynamics.
- Early stopping prevents overfitting; `best_snn_model.pth` is updated when validation accuracy improves.
