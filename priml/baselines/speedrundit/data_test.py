"""Processed ImageNet pairing, corpus receipts, and reference sampler order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

from PIL import Image

import numpy as np
import pytest
import torch

from priml.baselines.speedrundit.corpus import (
    CorpusMismatchError,
    save_stored,
    save_table,
    write_receipt,
)
from priml.baselines.speedrundit.data import (
    PairedImageLatentDataset,
    SpeedrunImageNetData,
)
from priml.baselines.speedrundit.latent_codec import FloatCodec, ScalarTableCodec
from priml.model.vision_ae.invae import INVAE


if TYPE_CHECKING:
    from pathlib import Path


def _source(root: Path, **fields: object) -> PairedImageLatentDataset.Config:
    """Return a source over ``root`` whose INVAE yields 2x2 latents (32px images)."""
    config = PairedImageLatentDataset.Config(working_dir=root)
    config.autoencoder = INVAE.Config(image_size=32)
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def _prepared_pair(
    root: Path, name: str, label: int, *, subdir: str = "vae-in"
) -> None:
    image_dir = root / "images" / "00000"
    latent_dir = root / subdir / "00000"
    image_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(label, 0, 0)).save(image_dir / f"img{name}.png")
    np.save(
        latent_dir / f"img-latents-{name}.npy",
        np.full((1, 32, 2, 2), label, dtype=np.float32),
    )


def _corpus(root: Path, config: PairedImageLatentDataset.Config, count: int) -> None:
    """Write ``count`` float32 pairs, their labels, and a matching receipt."""
    labels: list[list[str | int]] = []
    for index in range(count):
        name = f"{index:08d}"
        _prepared_pair(root, name, index + 5, subdir=config.latent_subdir)
        labels.append([f"00000/img-latents-{name}.npy", index + 5])
    latent_dir = root / config.latent_subdir
    (latent_dir / "dataset.json").write_text(json.dumps({"labels": labels}))
    write_receipt(
        latent_dir,
        autoencoder=config.autoencoder,
        codec_config=config.codec,
        codec=config.codec.make(),
        table_sha256=None,
        details={},
    )


def test_reference_names_and_tensor_values(tmp_path: Path) -> None:
    _corpus(tmp_path, _source(tmp_path), 10)
    dataset = _source(tmp_path).make()
    assert len(dataset) == 10
    assert torch.equal(dataset[0]["latent"], torch.full((32, 2, 2), 5.0))
    assert dataset[0]["image"][0, 0, 0] == 5
    assert dataset[0]["label"] == 5

    config = SpeedrunImageNetData.Config()
    config.source = _source(tmp_path)
    config.batch_size = 2
    config.num_workers = 0
    config.pin_memory = False
    data = config.make()
    torch.manual_seed(0)
    order = [batch["label"].tolist() for batch in data.train_dataloader()]
    assert order == [[11, 12], [6, 9], [7, 5], [14, 13], [8, 10]]


def test_a_corpus_without_a_receipt_is_refused(tmp_path: Path) -> None:
    _corpus(tmp_path, _source(tmp_path), 2)
    (tmp_path / "vae-in" / "corpus.json").unlink()
    with pytest.raises(FileNotFoundError, match="receipt-only"):
        _ = _source(tmp_path).make()


def test_a_corpus_from_another_autoencoder_is_refused(tmp_path: Path) -> None:
    """A receipt naming other weights fails, and the message names the field."""
    _corpus(tmp_path, _source(tmp_path), 2)
    autoencoder = INVAE.Config(image_size=32)
    autoencoder.checkpoint = None
    config = _source(tmp_path, autoencoder=autoencoder)
    with pytest.raises(CorpusMismatchError, match=r"autoencoder\.checkpoints"):
        _ = config.make()


def test_a_corpus_from_another_codec_is_refused(tmp_path: Path) -> None:
    _corpus(tmp_path, _source(tmp_path), 2)
    config = _source(tmp_path, codec=FloatCodec.Config(dtype=torch.float16))
    with pytest.raises(CorpusMismatchError, match=r"codec\.stored_dtype"):
        _ = config.make()


def test_a_latent_of_the_wrong_geometry_is_refused(tmp_path: Path) -> None:
    config = _source(tmp_path)
    config.autoencoder = INVAE.Config(image_size=64)
    _corpus(tmp_path, config, 1)
    dataset = config.make()
    with pytest.raises(ValueError, match="autoencoder produces"):
        _ = dataset[0]


def test_corpora_share_images_but_not_latents(tmp_path: Path) -> None:
    """A second corpus beside the first reuses its images and may cover fewer."""
    _corpus(tmp_path, _source(tmp_path), 3)
    other = _source(tmp_path, latent_subdir="other")
    _corpus(tmp_path, other, 2)
    assert len(other.make()) == 2
    assert len(_source(tmp_path).make()) == 3


def test_a_uint8_corpus_decodes_through_its_table(tmp_path: Path) -> None:
    """Stored indices come back as the fitted table's levels."""
    source = _source(tmp_path)
    source.codec = ScalarTableCodec.Config(num_fit_images=1)
    codec = source.codec.make()
    sample = torch.randn(4, 32, 2, 2, generator=torch.Generator().manual_seed(0))
    codec.fit(sample)
    latent_dir = tmp_path / "vae-in"
    (latent_dir / "00000").mkdir(parents=True)
    (tmp_path / "images" / "00000").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(tmp_path / "images/00000/img00000000.png")
    save_stored(latent_dir / "00000/img-latents-00000000.npy", codec.encode(sample[:1]))
    (latent_dir / "dataset.json").write_text(
        json.dumps({"labels": [["00000/img-latents-00000000.npy", 1]]}),
    )
    write_receipt(
        latent_dir,
        autoencoder=source.autoencoder,
        codec_config=source.codec,
        codec=codec,
        table_sha256=save_table(latent_dir, codec),
        details={},
    )
    latent = source.make()[0]["latent"]
    assert latent.dtype == torch.float32
    assert torch.equal(latent, codec.decode(codec.encode(sample[0])))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
