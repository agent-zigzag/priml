"""Latent normalizers keep each reference's exact operation order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from priml.model.vision_ae.checkpoint import LocalFile
from priml.model.vision_ae.custom_types import LatentNormalizer
from priml.model.vision_ae.latent_norm import (
    ChannelLatentStats,
    ElementwiseLatentStats,
    ScaleLatents,
)


if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


def _latent() -> Tensor:
    return torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(0))


def _stats_file(tmp_path: Path, **stats: Tensor | None) -> LocalFile.Config:
    path = tmp_path / "stats.pt"
    torch.save(stats, path)
    return LocalFile.Config(path=path)


def test_every_normalizer_satisfies_the_protocol(tmp_path: Path) -> None:
    stats = _stats_file(
        tmp_path, mean=torch.zeros(1, 3, 1, 1), std=torch.ones(1, 3, 1, 1)
    )
    assert isinstance(ScaleLatents.Config().make(), LatentNormalizer)
    assert isinstance(ChannelLatentStats.Config(stats=stats).make(), LatentNormalizer)


def test_scale_is_one_multiply_each_way() -> None:
    norm = ScaleLatents.Config(scale=0.3099).make()
    latent = _latent()
    assert torch.equal(norm.normalize(latent), latent * 0.3099)
    assert torch.equal(norm.denormalize(latent), latent / 0.3099)


def test_scale_refuses_zero() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        _ = ScaleLatents.Config(scale=0.0).make()


def test_elementwise_stats_follow_rae_order(tmp_path: Path) -> None:
    """``(z - mean) / sqrt(var + eps)`` and ``z * sqrt(var + eps) + mean``."""
    mean = torch.rand(3, 4, 4, generator=torch.Generator().manual_seed(1))
    var = torch.rand(3, 4, 4, generator=torch.Generator().manual_seed(2)) + 0.1
    config = ElementwiseLatentStats.Config(
        stats=_stats_file(tmp_path, mean=mean, var=var)
    )
    norm = config.make()
    latent = _latent()
    assert torch.equal(norm.normalize(latent), (latent - mean) / torch.sqrt(var + 1e-5))
    assert torch.equal(norm.denormalize(latent), latent * torch.sqrt(var + 1e-5) + mean)


def test_elementwise_stats_skip_a_missing_mean(tmp_path: Path) -> None:
    """RAE's 256px statistics carry no mean; nothing is subtracted, not zero."""
    var = torch.full((3, 4, 4), 4.0)
    config = ElementwiseLatentStats.Config(
        stats=_stats_file(tmp_path, mean=None, var=var)
    )
    norm = config.make()
    latent = _latent()
    assert torch.equal(norm.normalize(latent), latent / torch.sqrt(var + 1e-5))


def test_channel_stats_follow_lightningdit_order(tmp_path: Path) -> None:
    """``(z - mean) / std * m`` and ``z * std / m + mean``."""
    mean = torch.tensor([0.1, -0.2, 0.3]).view(1, 3, 1, 1)
    std = torch.tensor([0.08, 0.2, 0.13]).view(1, 3, 1, 1)
    config = ChannelLatentStats.Config(
        stats=_stats_file(tmp_path, mean=mean, std=std),
        multiplier=1.5,
    )
    norm = config.make()
    latent = _latent()
    assert torch.equal(norm.normalize(latent), (latent - mean) / std * 1.5)
    assert torch.equal(norm.denormalize(latent), (latent * std) / 1.5 + mean)


def test_statistics_files_are_required() -> None:
    with pytest.raises(ValueError, match="stats file"):
        _ = ElementwiseLatentStats.Config().make()
    with pytest.raises(ValueError, match="stats file"):
        _ = ChannelLatentStats.Config().make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
