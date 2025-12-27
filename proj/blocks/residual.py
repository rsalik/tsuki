from ..models import Network
from . import Block, BlockSettings


@dataclass
class ResidualSettings(BlockSettings):
    """Settings for the Residual block."""

    in_channels: int
    out_channels: int
    stride: int = 2
    downsampler: Network | Block | None = None


@dataclass
class ResidualBlock(Block):
    """Residual block using PyTorch"""

    settings: ResidualSettings

    def __init__(self, settings: ResidualSettings):
        super().__init__(settings)

    def setup(self, internal_block, in_dim, out_dim=None):
        internal_block.main_model = nn.Sequential(
            nn.Conv2d(
                self.settings.in_channels,
                self.settings.out_channels,
                kernel_size=3,
                stride=self.settings.stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.settings.out_channels),
            nn.ReLU(),
            nn.Conv2d(
                self.settings.out_channels,
                self.settings.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.settings.out_channels),
        )

    def forward(self, internal_block, x: Tensor):
        id = x
        out = internal_block.main_model(x)

        if self.settings.downsampler is not None:
            id = self.settings.downsampler(id)

        return nn.functional.relu(id + out)

    @property
    def out_dim(self) -> int:
        return self.settings.hidden_dim
