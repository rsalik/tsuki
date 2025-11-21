from proj.envs.gym import GymEnv
from ..agents import ZeroAgent, MixedExplorationAgent
from ..envs.base import DummyEnv
from ..utils.printing import Task, progress
from ..models import Model
from ..blocks import MLPBlock, SpectralBlock, MLPSettings, SpectralSettings

import numpy as np
import matplotlib.pyplot as plt
import torch


def simple_train(
    model, epochs=10, batch_size=32, lr=0.001, datapoints=10000, holdout=2000
):
    
    blocks, model_name = model

    env = DummyEnv()
    agent = MixedExplorationAgent(env)
    model = Model.from_blocks(blocks, env)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = lambda preds, ys: torch.mean((preds - ys) ** 2)

    state, obs = env.reset()
    data = [
        {
            "next_obs": obs["value"],
            "previous": obs["value"],
            "action": np.zeros(env.action_dim),
        }
    ]

    for i in range(datapoints):
        act = agent.act(obs)
        state, next_obs = env.step(state, act)
        data.append(
            {"next_obs": next_obs["value"], "previous": obs["value"], "action": act}
        )

        obs = next_obs

    train_data = data[: datapoints - holdout]
    test_data = data[datapoints - holdout :]

    # Before training - prepare data as tensors
    train_X = torch.from_numpy(
        np.array([np.concatenate([d["previous"], d["action"]]) for d in train_data])
    ).float()
    train_Y = torch.from_numpy(np.array([d["next_obs"] for d in train_data])).float()
    test_X = torch.from_numpy(
        np.array([np.concatenate([d["previous"], d["action"]]) for d in test_data])
    ).float()
    test_Y = torch.from_numpy(np.array([d["next_obs"] for d in test_data])).float()

    # Use DataLoader for batching
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_preds = []

    losses = []
    epoch_boundaries = []

    test_preds_untrained = []

    with torch.no_grad():
        test_preds_untrained = model.forward(test_X)

    with Task(f"Training {model_name} for {epochs} Epochs", epochs) as e_task:
        for epoch in range(epochs):
            model.reset()
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()  # Changed from model.zero_grad()

                preds = model.forward(x_batch)
                loss = loss_fn(preds, y_batch)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

            epoch_boundaries.append(len(losses))

            # Evaluate on test set
            model.eval()
            model.reset() 
            with torch.no_grad():  # Disable gradient computation
                test_preds = model.forward(test_X)
                avg_loss = loss_fn(test_preds, test_Y)

            e_task.update(text=f"Avg Test Loss: {avg_loss:.6f}")

    num_model_params = sum(p.numel() for p in model.parameters())
    return test_preds, test_Y, test_preds_untrained, losses, epoch_boundaries, num_model_params