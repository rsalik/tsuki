from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np


def _as_2d(arr: np.ndarray | Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
    """
    Ensure input is a 2D NumPy array of shape (N, D).

    - If 1D, reshape to (N, 1)
    - If >2D, flatten leading dims into N while preserving the last dim as D
    """
    a = np.asarray(arr)
    if a.ndim == 0:
        # scalar -> (1, 1)
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim > 2:
        return a.reshape(-1, a.shape[-1])
    return a  # already (N, D)


def pred_vs_true_plot(
    real_next_observations: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    predicted_next_observations: (
        np.ndarray | Sequence[float] | Sequence[Sequence[float]]
    ),
    ax: Optional[Any] = None,
    *,
    title: Optional[str] = None,
    dim_labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    marker: Any = "o",
    size: float = 20,
    alpha: float = 0.6,
    show_legend: bool = True,
) -> Tuple[Any | None, Any]:
    """
    Plot predicted vs. true next observations as a scatter plot.

    Parameters
    ----------
    real_next_observations:
            Array-like of shape (N, D) or (N,) with the true next observations.
    predicted_next_observations:
            Array-like of shape (N, D) or (N,) with the predicted next observations.
    ax:
            Optional matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title:
            Optional plot title. Defaults to "Predicted vs True".
    dim_labels:
            Optional per-dimension labels for the legend; length must equal D if provided.
    colors:
            Optional list of colors per dimension; length must equal D if provided.
    marker:
            Matplotlib marker for points (default "o").
    size:
            Marker size (default 20).
    alpha:
            Point alpha (default 0.6).
    show_legend:
            Whether to show a legend when D > 1 (default True).

    Returns
    -------
    (fig, ax):
            If `ax` is provided, returns (None, ax). Otherwise returns the created (fig, ax).
    """

    # Normalize inputs to (N, D)
    true = _as_2d(real_next_observations)
    pred = _as_2d(predicted_next_observations)

    if true.shape != pred.shape:
        raise ValueError(
            f"Shapes must match, got true {true.shape} vs pred {pred.shape}"
        )

    N, D = true.shape

    # Lazy import to avoid hard dependency until the function is used
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover - import-time environment issue
        raise ImportError(
            "matplotlib is required for pred_vs_true_plot; install with 'pip install matplotlib'"
        ) from e

    created_ax = False
    fig: Optional[Any] = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        created_ax = True

    # Setup colors/labels
    default_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    cols = list(colors) if colors is not None else list(default_colors)
    # Repeat if not enough colors
    if len(cols) < D:
        reps = int(np.ceil(D / max(1, len(cols))))
        cols = (cols * reps)[:D]

    labels = dim_labels if dim_labels is not None else [f"y_{i}" for i in range(D)]

    # Scatter per dimension for clarity
    for i in range(D):
        ax.scatter(
            true[:, i],
            pred[:, i],
            s=size,
            alpha=alpha,
            color=cols[i],
            label=labels[i] if (show_legend and D > 1) else None,
            marker=marker,
            edgecolors="none",
        )

    # Diagonal y=x reference line covering data range with small padding
    data_min = np.nanmin([true.min(), pred.min()])
    data_max = np.nanmax([true.max(), pred.max()])
    if np.isfinite([data_min, data_max]).all():
        span = data_max - data_min
        pad = 0.05 * span if span > 0 else 1.0
        lo, hi = data_min - pad, data_max + pad
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6, linewidth=1.0, label="y=x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    else:
        # Fallback: simple unit diagonal
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1.0, label="y=x")

    ax.set_xlabel("True next observations")
    ax.set_ylabel("Predicted next observations")
    ax.set_title(title or "Predicted vs True")
    if show_legend and D > 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    return (fig if created_ax else None, ax)


def losses_plot(
    train_losses: Sequence[float],
    epoch_boundaries: Optional[Sequence[int]] = None,
    ax: Optional[Any] = None,
    *,
    title: Optional[str] = None,
) -> Tuple[Any | None, Any]:
    # Lazy import to avoid hard dependency until the function is used
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover - import-time environment issue
        raise ImportError(
            "matplotlib is required for pred_vs_true_plot; install with 'pip install matplotlib'"
        ) from e

    created_ax = False
    fig: Optional[Any] = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        created_ax = True

    ax.plot(train_losses, label="Train Loss")

    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")

    if epoch_boundaries is not None:
        # Add vertical lines for epoch boundaries
        for epoch_end in epoch_boundaries:
            ax.axvline(x=epoch_end, color="gray", linestyle="--", alpha=0.5)

    ax.set_title(title or "Training Loss Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return (fig if created_ax else None, ax)


def multi_model_losses_plot(
    models_data: Sequence[Tuple[Sequence[float], str]],
    epoch_boundaries: Optional[Sequence[int]] = None,
    ax: Optional[Any] = None,
    *,
    title: Optional[str] = None,
) -> Tuple[Any | None, Any]:
    """
    Plot training losses for multiple models on the same axes.

    Parameters
    ----------
    models_data:
            List of tuples, where each tuple is (train_losses, model_name).
    epoch_boundaries:
            Optional list of batch indices marking epoch boundaries.
    ax:
            Optional matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title:
            Optional plot title. Defaults to "Training Loss Over Time - Multiple Models".

    Returns
    -------
    (fig, ax):
            If `ax` is provided, returns (None, ax). Otherwise returns the created (fig, ax).
    """
    # Lazy import to avoid hard dependency until the function is used
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover - import-time environment issue
        raise ImportError(
            "matplotlib is required for multi_model_losses_plot; install with 'pip install matplotlib'"
        ) from e

    created_ax = False
    fig: Optional[Any] = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        created_ax = True

    # Default colors for different models
    default_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    
    # Plot each model's losses with a different color
    for idx, (train_losses, model_name) in enumerate(models_data):
        color = default_colors[idx % len(default_colors)]
        ax.plot(train_losses, label=model_name, color=color)

    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")

    if epoch_boundaries is not None:
        # Add vertical lines for epoch boundaries
        for epoch_end in epoch_boundaries:
            ax.axvline(x=epoch_end, color="gray", linestyle="--", alpha=0.5)

    ax.set_title(title or "Training Loss Over Time - Multiple Models")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return (fig if created_ax else None, ax)


def multi_model_pred_vs_true_plot(
    models_data: Sequence[Tuple[np.ndarray | Sequence[float] | Sequence[Sequence[float]], 
                                 np.ndarray | Sequence[float] | Sequence[Sequence[float]], 
                                 str]],
    ax: Optional[Any] = None,
    *,
    title: Optional[str] = None,
    marker: Any = "o",
    size: float = 20,
    alpha: float = 0.6,
    show_legend: bool = True,
) -> Tuple[Any | None, Any]:
    """
    Plot predicted vs. true next observations for multiple models on the same axes.

    Parameters
    ----------
    models_data:
            List of tuples, where each tuple is (real_next_observations, predicted_next_observations, model_name).
            Each observation array should be of shape (N, D) or (N,).
    ax:
            Optional matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title:
            Optional plot title. Defaults to "Predicted vs True - Multiple Models".
    marker:
            Matplotlib marker for points (default "o").
    size:
            Marker size (default 20).
    alpha:
            Point alpha (default 0.6).
    show_legend:
            Whether to show a legend (default True).

    Returns
    -------
    (fig, ax):
            If `ax` is provided, returns (None, ax). Otherwise returns the created (fig, ax).
    """
    # Lazy import to avoid hard dependency until the function is used
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover - import-time environment issue
        raise ImportError(
            "matplotlib is required for multi_model_pred_vs_true_plot; install with 'pip install matplotlib'"
        ) from e

    created_ax = False
    fig: Optional[Any] = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        created_ax = True

    # Default colors for different models
    default_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    # Track data bounds across all models for the diagonal line
    all_data_min = float('inf')
    all_data_max = float('-inf')

    # Plot each model's predictions with a different color
    for idx, (pred_next_obs, real_next_obs, model_name) in enumerate(models_data):
        # Normalize inputs to (N, D)
        true = _as_2d(real_next_obs)
        pred = _as_2d(pred_next_obs)

        if true.shape != pred.shape:
            raise ValueError(
                f"Model '{model_name}': shapes must match, got true {true.shape} vs pred {pred.shape}"
            )

        color = default_colors[idx % len(default_colors)]

        # Flatten all dimensions for this model and plot together
        true_flat = true.flatten()
        pred_flat = pred.flatten()

        ax.scatter(
            true_flat,
            pred_flat,
            s=size,
            alpha=alpha,
            color=color,
            label=model_name if show_legend else None,
            marker=marker,
            edgecolors="none",
        )

        # Update data bounds
        all_data_min = min(all_data_min, np.nanmin(true), np.nanmin(pred))
        all_data_max = max(all_data_max, np.nanmax(true), np.nanmax(pred))

    # Diagonal y=x reference line covering data range with small padding
    if np.isfinite([all_data_min, all_data_max]).all():
        span = all_data_max - all_data_min
        pad = 0.05 * span if span > 0 else 1.0
        lo, hi = all_data_min - pad, all_data_max + pad
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6, linewidth=1.0, label="y=x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    else:
        # Fallback: simple unit diagonal
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1.0, label="y=x")

    ax.set_xlabel("True next observations")
    ax.set_ylabel("Predicted next observations")
    ax.set_title(title or "Predicted vs True - Multiple Models")
    if show_legend:
        ax.legend()
    ax.grid(True, alpha=0.3)

    return (fig if created_ax else None, ax)