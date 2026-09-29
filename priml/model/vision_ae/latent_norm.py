"""Latent normalizers: raw autoencoder latents to the space a diffusion model sees.

One class per published convention, each in its reference's exact operation
order. They are NOT folded into one affine map: ``sqrt(var + eps)`` is not the
stored ``std``, and multiplying by a reciprocal is not dividing, so a merged
form would move the last bits every golden pins.

A normalizer is separate from the autoencoder (which never applies it) and from
a storage codec (which never sees it): a corpus stores raw latents, so changing
normalization never requires re-encoding.

Normalizers are plain objects, not modules: they hold no parameters and run on
data rather than in a model slot. Statistics follow the latent to its device on
first use, as the reference moves them per call, and are cached there.
"""

from __future__ import annotations

from typing import cast

from configgle import Fig, Makeable
from torch import Tensor

import torch

from priml.model.vision_ae.custom_types import CheckpointFile


class ScaleLatents:
    """Multiply by one scalar: ``z * scale`` and back by ``z / scale``."""

    class Config(Fig["ScaleLatents"]):
        """The scalar."""

        scale: float = 1.0
        """Multiplier into diffusion space; INVAE publishes 0.3099."""

    def __init__(self, config: Config) -> None:
        if config.scale == 0:
            raise ValueError("ScaleLatents needs a nonzero scale.")
        self.scale = config.scale

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent * scale``.

        Args:
          latent: Raw latent.

        Returns:
          normalized: Diffusion-space latent, in ``latent``'s dtype.

        """
        return latent * self.scale

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent / scale``.

        Args:
          latent: Diffusion-space latent.

        Returns:
          raw: Raw latent, in ``latent``'s dtype.

        """
        return latent / self.scale


# A base class, so it precedes the two normalizers that share it.
class _DeviceStats:
    """Statistics kept on the CPU and copied, once per device, to where latents are."""

    def __init__(self, stats: dict[str, Tensor | None]) -> None:
        self._stats: dict[torch.device, dict[str, Tensor | None]] = {
            torch.device("cpu"): stats,
        }

    def _on(self, latent: Tensor) -> dict[str, Tensor | None]:
        """Return the statistics on ``latent``'s device."""
        device = latent.device
        if device not in self._stats:
            source = self._stats[torch.device("cpu")]
            self._stats[device] = {
                name: None if value is None else value.to(device)
                for name, value in source.items()
            }
        return self._stats[device]


class ElementwiseLatentStats(_DeviceStats):
    """Standardize each ``[C, H, W]`` element: ``(z - mean) / sqrt(var + eps)``.

    RAE's convention, with statistics estimated over the training images and
    stored as ``{"mean": Tensor | None, "var": Tensor | None}``; a ``None``
    member is skipped, as the reference skips it, rather than replaced by 0 or 1
    -- subtracting a zero tensor is exact, but dividing by ``sqrt(1 + eps)`` is
    not the identity.

    References:
      https://github.com/bytetriper/RAE/blob/a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041/src/stage1/rae.py

    """

    class Config(Fig["ElementwiseLatentStats"]):
        """Where the statistics live."""

        stats: Makeable[CheckpointFile] | None = None
        """File holding ``mean`` and ``var``."""

        eps: float = 1e-5
        """Added to the variance before the square root."""

    def __init__(self, config: Config) -> None:
        if config.stats is None:
            raise ValueError("ElementwiseLatentStats needs a stats file.")
        payload = cast(
            "dict[str, Tensor | None]",
            torch.load(
                config.stats.make().path(), map_location="cpu", weights_only=True
            ),
        )
        super().__init__({"mean": payload.get("mean"), "var": payload.get("var")})
        self.eps = config.eps

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Return ``(latent - mean) / sqrt(var + eps)``.

        Args:
          latent: ``[B, C, H, W]`` raw latent.

        Returns:
          normalized: Standardized latent.

        """
        stats = self._on(latent)
        mean, var = stats["mean"], stats["var"]
        if mean is not None:
            latent = latent - mean
        if var is not None:
            latent = latent / torch.sqrt(var + self.eps)
        return latent

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent * sqrt(var + eps) + mean``.

        Args:
          latent: ``[B, C, H, W]`` standardized latent.

        Returns:
          raw: Raw latent.

        """
        stats = self._on(latent)
        mean, var = stats["mean"], stats["var"]
        if var is not None:
            latent = latent * torch.sqrt(var + self.eps)
        if mean is not None:
            latent = latent + mean
        return latent


class ChannelLatentStats(_DeviceStats):
    """Standardize per channel, then scale: ``(z - mean) / std * multiplier``.

    VTP's convention, applied by its LightningDiT training data and inverted as
    ``z * std / multiplier + mean`` before decoding. Statistics are stored as
    ``{"mean": [1, C, 1, 1], "std": [1, C, 1, 1]}``.

    References:
      https://github.com/JingfengYao/LightningDiT/blob/986098540d7d902cfee84e5344804de0eeeb2aaa/datasets/img_latent_dataset.py

    """

    class Config(Fig["ChannelLatentStats"]):
        """Where the statistics live, and the multiplier after them."""

        stats: Makeable[CheckpointFile] | None = None
        """File holding ``mean`` and ``std``."""

        multiplier: float = 1.0
        """Applied after standardizing; VTP's configs set 1.0."""

    def __init__(self, config: Config) -> None:
        if config.stats is None:
            raise ValueError("ChannelLatentStats needs a stats file.")
        payload = cast(
            "dict[str, Tensor]",
            torch.load(
                config.stats.make().path(), map_location="cpu", weights_only=True
            ),
        )
        super().__init__({"mean": payload["mean"], "std": payload["std"]})
        self.multiplier = config.multiplier

    def normalize(self, latent: Tensor, /) -> Tensor:
        """Return ``(latent - mean) / std * multiplier``.

        Args:
          latent: ``[B, C, H, W]`` raw latent.

        Returns:
          normalized: Standardized, scaled latent.

        """
        mean, std = self._stats_on(latent)
        return (latent - mean) / std * self.multiplier

    def denormalize(self, latent: Tensor, /) -> Tensor:
        """Return ``latent * std / multiplier + mean``.

        Args:
          latent: ``[B, C, H, W]`` diffusion-space latent.

        Returns:
          raw: Raw latent.

        """
        mean, std = self._stats_on(latent)
        return (latent * std) / self.multiplier + mean

    def _stats_on(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """Return the mean and std on ``latent``'s device; both are always stored."""
        stats = self._on(latent)
        mean, std = stats["mean"], stats["std"]
        assert mean is not None
        assert std is not None
        return mean, std
