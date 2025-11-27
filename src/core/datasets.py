"""Dataset abstractions for BinaryLLM Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from src.core import dataset_catalog
from src.core.embeddings import FloatEmbeddingBatch


def _resolve_dataset(name: str) -> dataset_catalog.DatasetSpec:
    try:
        return dataset_catalog.get_dataset_spec(name)
    except dataset_catalog.UnknownDatasetError as exc:
        raise ValueError(f"dataset '{name}' is not registered") from exc


def _resolve_encoder(name: str) -> dataset_catalog.EncoderSpec:
    try:
        return dataset_catalog.get_encoder_spec(name)
    except dataset_catalog.UnknownEncoderError as exc:
        raise ValueError(f"encoder '{name}' is not registered") from exc


@dataclass(slots=True)
class EmbeddingDataset:
    """Holds embeddings plus optional labels/splits with catalog-driven validation."""

    name: str
    encoder_name: Optional[str]
    embeddings: FloatEmbeddingBatch
    labels: Optional[Sequence[str]] = None
    texts: Optional[Sequence[str]] = None
    split: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        spec = _resolve_dataset(self.name)

        effective_encoder = self.encoder_name or spec.default_encoder
        if effective_encoder is None:
            raise ValueError(f"dataset '{self.name}' requires an explicit encoder_name")
        _resolve_encoder(effective_encoder)
        self.encoder_name = effective_encoder

        if self.embeddings.embedding_dim != spec.embedding_dim:
            raise ValueError("embedding_dim mismatch for embeddings (stage='embeddings').")

        if spec.requires_l2_normalization and not self.embeddings.normalized:
            raise ValueError("normalized embeddings are required for this dataset (stage='embeddings').")

        n_items = self.embeddings.data.shape[0]

        if self.labels is not None:
            if len(self.labels) != n_items:
                raise ValueError("labels length must match embeddings (stage='embeddings').")
            if not spec.supports_classification:
                raise ValueError(f"classification labels are not supported for dataset '{self.name}'.")
            self.labels = list(self.labels)

        if self.split is not None:
            if len(self.split) != n_items:
                raise ValueError("split length must match embeddings (stage='embeddings').")
            self.split = list(self.split)

        if self.texts is not None:
            if len(self.texts) != n_items:
                raise ValueError("texts length must match embeddings (stage='embeddings').")
            self.texts = list(self.texts)


@dataclass(slots=True)
class QueryDataset:
    """Represents query embeddings targeting a specific dataset."""

    queries: FloatEmbeddingBatch
    target_dataset_name: str

    def __post_init__(self) -> None:
        try:
            dataset_catalog.get_dataset_spec(self.target_dataset_name)
        except dataset_catalog.UnknownDatasetError as exc:
            raise ValueError(
                f"target_dataset '{self.target_dataset_name}' is not registered"
            ) from exc

