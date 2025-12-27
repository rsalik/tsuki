from ..blocks.base import Block, IntBlock

# from blocks.history import HistoryWrapper, HistoryWrapperSettings  # removed: using a torch-native utility instead
from ..envs.base import Env

import torch
from torch import nn, Tensor
from typing import Any, List


class HistoryConcat(nn.Module):
    """
    Torch-native history concatenation utility.

    Semantics match the old HistoryWrapper:
    - Keeps the last `history_length` outputs of the previous block, including the current x.
    - If not enough past items, pads with zeros at the front.
    - Concatenates along feature dimension for batched inputs [B, D] -> [B, D*k],
      or along the only dimension for vectors [D] -> [D*k].
    - Past features are stored detached (no BPTT), current x remains attached for gradients.

    Note: If batch size changes between calls, the buffer is reset.
    """

    def __init__(self, in_dim: int, history_length: int):
        super().__init__()
        self.in_dim = in_dim
        self.history_length = int(history_length)
        self.out_dim = in_dim * self.history_length
        # Python list buffer; we only store detached tensors here
        self._buffer: List[Tensor] = []
        self._last_batch: int | None = None

    @torch.no_grad()
    def reset(self):
        self._buffer.clear()
        self._last_batch = None

    def _pad_and_stack(self, items: List[Tensor], like: Tensor) -> Tensor:
        # items already sized to <= history_length; pad at front with zeros to reach exactly history_length
        needed = self.history_length - len(items)
        if needed > 0:
            if like.dim() == 2:
                pad = like.new_zeros(
                    needed, like.size(0), like.size(1)
                )  # [needed, B, D]
            elif like.dim() == 1:
                pad = like.new_zeros(needed, like.size(0))  # [needed, D]
            else:
                raise ValueError(
                    f"HistoryConcat expects 1D or 2D tensors, got shape {tuple(like.shape)}"
                )
            items = [*pad, *items]  # type: ignore[list-item]
        # Now select the last exactly history_length
        items = items[-self.history_length :]

        if like.dim() == 2:
            # items: list of [B, D] -> cat along last dim to [B, D*k]
            return torch.cat(items, dim=-1)
        else:
            # items: list of [D] -> cat along dim 0 to [D*k]
            return torch.cat(items, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() not in (1, 2):
            raise ValueError(f"HistoryConcat expects 1D or 2D input, got {x.dim()}D")

        # Reset buffer if batch size changed (for 2D inputs)
        if x.dim() == 2:
            bsz, d = x.shape
            if d != self.in_dim:
                raise ValueError(
                    f"HistoryConcat expected feature dim {self.in_dim}, got {d}"
                )
            if self._last_batch is None:
                self._last_batch = bsz
            elif self._last_batch != bsz:
                # batch size changed; reset to avoid shape mismatch
                self.reset()
                self._last_batch = bsz
        else:
            if x.numel() != self.in_dim:
                raise ValueError(
                    f"HistoryConcat expected vector of dim {self.in_dim}, got {x.numel()}"
                )

        # Compose a history list including current x, with past entries detached
        past = [t.detach() for t in self._buffer]
        hist_with_current = [*past, x]  # current x kept attached

        # Build output by padding/concatenation
        y = self._pad_and_stack(hist_with_current, like=x)

        # Update buffer for the next call (store detached current x)
        with torch.no_grad():
            self._buffer.append(x.detach())
            if len(self._buffer) > self.history_length:
                self._buffer.pop(0)

        return y


class _IntBlockAdapter(nn.Module):
    """
    Adapter to make a non-nn.Module IntBlock look like an nn.Module,
    and to register any nested torch Modules/Parameters so optimizers can find them.
    """

    def __init__(self, int_block: IntBlock):
        super().__init__()
        self.block = int_block
        # Expose out_dim if present
        self.out_dim = getattr(int_block, "out_dim", None)

        # Adopt child modules/parameters found on the int_block so that:
        # - they are included in model.parameters()
        # - they move with .to(device)
        # We do a shallow scan of attributes + container items.
        self._adopt_children(int_block)

    def _adopt_children(self, obj: Any, prefix: str = "") -> None:
        try:
            items = list(vars(obj).items())
        except TypeError:
            items = []

        def reg_param(name: str, p: nn.Parameter):
            # Avoid name collisions
            safe_name = f"_wrapped_param__{name}"
            if not hasattr(self, safe_name):
                self.register_parameter(safe_name, p)

        def reg_module(name: str, m: nn.Module):
            safe_name = f"_wrapped_mod__{name}"
            if not hasattr(self, safe_name):
                setattr(self, safe_name, m)

        for name, val in items:
            if isinstance(val, nn.Parameter):
                reg_param(f"{prefix}{name}", val)
            elif isinstance(val, nn.Module):
                reg_module(f"{prefix}{name}", val)
            elif isinstance(val, (list, tuple)):
                # Register individual modules/parameters within containers
                for i, v in enumerate(val):
                    n = f"{prefix}{name}_{i}"
                    if isinstance(v, nn.Module):
                        reg_module(n, v)
                    elif isinstance(v, nn.Parameter):
                        reg_param(n, v)
            elif isinstance(val, dict):
                for k, v in val.items():
                    n = f"{prefix}{name}_{k}"
                    if isinstance(v, nn.Module):
                        reg_module(n, v)
                    elif isinstance(v, nn.Parameter):
                        reg_param(n, v)
            # Note: we keep this shallow to avoid accidental deep recursion into arbitrary objects.

    def forward(self, x: Tensor) -> Tensor:
        return self.block.forward(x)


class Network:
    """A network is simply a wrapper for a sequence of blocks."""

    blocks: list[Block]

    def __init__(self, blocks: list[Block]):
        self.blocks = blocks


class Model(nn.Module):
    """A model composed of a sequence of blocks."""

    internal_blocks: nn.ModuleList
    env_name: str

    def __init__(self, internal_blocks: list[IntBlock], env_name: str):
        super().__init__()
        self.env_name = env_name

        modules: List[nn.Module] = []
        for b in internal_blocks:
            if isinstance(b, nn.Module):
                modules.append(b)
            else:
                modules.append(_IntBlockAdapter(b))
        self.internal_blocks = nn.ModuleList(modules)

    @classmethod
    def from_blocks(cls, blocks: list[Block | Network], env: Env) -> "Model":
        """Create a model from a list of blocks."""
        internal_blocks: list[IntBlock | nn.Module] = []
        current_in_dim = env.obs_dim + env.action_dim

        # Convert networks to blocks
        true_blocks: list[Block] = []
        for block in blocks:
            if isinstance(block, Network):
                true_blocks.extend(block.blocks)
            else:
                true_blocks.append(block)

        for i, block in enumerate(true_blocks):
            # Last block must output in the prediction space
            out_dim = None
            if i == len(true_blocks) - 1:
                if getattr(block, "_block_not_last", False):
                    raise ValueError("Last block in model cannot be a not_last block.")
                out_dim = env.obs_dim

            # If the block needs history, insert a torch-native history feature block first
            if getattr(block, "_block_takes_history", False):
                history_length = int(getattr(block, "history_length"))
                hist_module = HistoryConcat(
                    in_dim=current_in_dim, history_length=history_length
                )
                internal_blocks.append(hist_module)
                current_in_dim = hist_module.out_dim  # type: ignore[attr-defined]

            # Create and setup the block's internal implementation
            internal_block = block.create(current_in_dim)
            block.setup(internal_block, current_in_dim, out_dim)
            internal_blocks.append(internal_block)
            # Propagate dimensionality for the next block
            next_out_dim = getattr(internal_block, "out_dim", None)
            if next_out_dim is None:
                raise AttributeError(
                    f"Internal block {type(internal_block).__name__} must define out_dim"
                )
            current_in_dim = next_out_dim

        model = cls(internal_blocks, env.name)  # type: ignore
        return model

    def forward(self, x: Tensor) -> Tensor:
        for internal_block in self.internal_blocks:
            x = internal_block(x)  # works for nn.Module and adapter
        return x

    def reset(self) -> None:
        """Reset any internal stateful blocks (e.g., history buffers)."""
        for internal_block in self.internal_blocks:
            if hasattr(internal_block, "reset") and callable(internal_block.reset):
                internal_block.reset()
