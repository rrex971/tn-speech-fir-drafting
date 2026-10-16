from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    seed: int = 42
    encoder: str = 'law-ai/InLegalBERT'
    encoder_revision: str | None = None
    freeze_encoder: bool = False
    feature_batch_size: int = 16
    protocol: str = 'canonical'
    bank_initialization: str = 'decomposed'
    bank_size: int = 256
    dimension: int = 256
    groups: int = 4
    norm: str = 'godel'
    variant: str = 'and_or'
    pooling: str = 'mean'
    truncation: str = '512'
    max_length: int = 512
    statute_length: int = 256
    max_chunks: int = 8
    batch_size: int = 2
    accumulation: int = 8
    epochs: int = 5
    learning_rate: float = 2e-5
    loss: str = 'bce'
    grounding_weight: float = 0.1
    anchor_weight: float = 0.1
    sparsity_weight: float = 0.001
    anchor_fraction: float = 0.8
    target_risk: float = 0.1
    device: str = 'cpu'

    def validate(self) -> None:
        if self.norm not in {'godel', 'product', 'lukasiewicz'}:
            raise ValueError('Unknown t-norm')
        if self.protocol not in {'canonical', 'supplement'}:
            raise ValueError('Unknown dataset protocol')
        if self.bank_initialization not in {'decomposed', 'grounded'}:
            raise ValueError('Unknown bank initialization')
        if self.protocol == 'canonical' and self.bank_initialization != 'decomposed':
            raise ValueError('Canonical experiments require the complete P2 decomposition cache')
        if self.variant not in {'flat', 'conjunction', 'and_or', 'per_section', 'handwritten'}:
            raise ValueError('Unknown model variant')
        if self.pooling not in {'cls', 'mean', 'attention'} or self.truncation not in {'512', 'chunked'}:
            raise ValueError('Unknown encoder ablation')
        if self.loss not in {'bce', 'pos_weight', 'focal', 'asymmetric'}:
            raise ValueError('Unknown loss')
        if not 0 <= self.anchor_fraction < 1 or not 0 <= self.target_risk <= 1:
            raise ValueError('Invalid calibration or anchor fraction')
        if min(self.bank_size, self.dimension, self.groups, self.batch_size, self.accumulation, self.epochs,
               self.feature_batch_size, self.max_chunks) < 1:
            raise ValueError('Model and training dimensions must be positive')

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> Config:
    values = json.loads(Path(path).read_text()) if path else {}
    result = Config(**values)
    result.validate()
    return result


def data_dir() -> Path:
    return Path(os.environ.get('LEXFORM_DATA_DIR', str(ROOT / 'data' / 'ilsi'))).expanduser()


def output_dir() -> Path:
    return Path(os.environ.get('LEXFORM_OUTPUT_DIR', str(ROOT / 'artifacts' / 'lexform'))).expanduser()
