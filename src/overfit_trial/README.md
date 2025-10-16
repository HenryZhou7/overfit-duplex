# Overfit Duplex Model

## Prerequisite

### Data Preparation

Refer to `src/notebooks/mimi_feature_exploration.ipynb` to generate `*.npz` under `asset/single_pair_dataset`.

## Model Training

Run the following command to train the model
```
uv run python src/overfit_trial/train.py
```

Training logs will be stored under `runs/` and checkpoints will be stored under `checkpoints/`.
