def historical(cls):
    """Decorator to mark a Block as historical, meaning it takes as input a history of the past outputs of
    the previous block (or the environment state if it's the first block)."""

    cls._block_takes_history = True

    if not hasattr(cls, "history_length"):
        raise NotImplementedError(
            f"Class {cls.__name__} is marked as historical but does not implement "
            f"the required @property 'history_length'."
        )
    return cls


def not_last(cls):
    """Decorator to mark a Block as not being the last block in a model."""

    cls._block_not_last = True
    return cls


def conv(cls):
    """Decorator to mark a Block as treating input and output dimensions as channels."""
    cls._is_conv_block = True
    return cls
