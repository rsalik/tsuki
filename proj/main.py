import torch

from .utils.printing import Task, progress
from .tests.simple_training import simple_train
from .blocks import MLPBlock, MLPSettings, SpectralBlock, SpectralSettings
from .utils.plotting import multi_model_losses_plot, multi_model_pred_vs_true_plot, pred_vs_true_plot

import matplotlib.pyplot as plt

models = [
    ([
        MLPBlock(MLPSettings(out_dim=32)),
        MLPBlock(MLPSettings()),
    ], "Double MLP"),
    ([
        MLPBlock(MLPSettings(out_dim=8)),
        SpectralBlock(SpectralSettings(out_dim=8)),
        MLPBlock(MLPSettings()),
    ], "Sandwich"),
    ([
        MLPBlock(MLPSettings(out_dim=4)),
        SpectralBlock(SpectralSettings(out_dim=4)),
        MLPBlock(MLPSettings(out_dim=4)),
        SpectralBlock(SpectralSettings(out_dim=4)),
        MLPBlock(MLPSettings(out_dim=4)),
        SpectralBlock(SpectralSettings(out_dim=4)),
        MLPBlock(MLPSettings(out_dim=4)),
    ], "Club Club Sandwich"),
    ([
        MLPBlock(MLPSettings(out_dim=8)),
        MLPBlock(MLPSettings(out_dim=8)),
        SpectralBlock(SpectralSettings(out_dim=8)),
        MLPBlock(MLPSettings(out_dim=8)),
        MLPBlock(MLPSettings()),
    ], "Mostly Bread Sandwich"),
]

epoch_boundaries = []

all_test_preds = []
all_test_preds_untrained = []

all_losses = []

all_n_params = []

for model in progress(models, name=f"Training {len(models)} Models", keep_children=True):
    blocks, model_name = model

    test_preds, test_Y, test_preds_untrained, losses, epoch_boundaries, n_params = simple_train(model)
    all_test_preds.append((test_preds, test_Y, model_name))
    all_test_preds_untrained.append((test_preds_untrained, test_Y, model_name))
    all_losses.append((losses, model_name))
    all_n_params.append((n_params, model_name))

print(f"Trained {len(all_n_params)} models.")
for n_params, name in all_n_params:
    print(f"  -  {name:<30} {n_params:>10} params")

# Plot predictions after training
multi_model_losses_plot(
    all_losses,
    epoch_boundaries=epoch_boundaries,
    title="Training Losses for Different Models",
)

multi_model_pred_vs_true_plot(
    all_test_preds,
    title="Model Predictions vs True Values After Training",
)

plt.show()