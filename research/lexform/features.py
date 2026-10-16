from __future__ import annotations

import hashlib
import json
import sqlite3

import numpy as np

from .config import Config, output_dir


class FeatureStore:
    def __init__(self, cfg: Config, revision: str):
        signature = json.dumps({'encoder': cfg.encoder, 'revision': revision, 'pooling': cfg.pooling,
                               'truncation': cfg.truncation, 'length': cfg.max_length,
                               'statute_length': cfg.statute_length, 'chunks': cfg.max_chunks}, sort_keys=True)
        path = output_dir() / 'features' / f'{hashlib.sha256(signature.encode()).hexdigest()[:24]}.sqlite3'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute('CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, value BLOB NOT NULL)')

    @staticmethod
    def key(text: str, statute: bool) -> str:
        return hashlib.sha256((str(statute) + '\0' + text).encode()).hexdigest()

    def get(self, text: str, statute: bool):
        row = self.connection.execute('SELECT value FROM features WHERE key=?', (self.key(text, statute),)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32).copy() if row else None

    def put(self, text: str, statute: bool, values: np.ndarray) -> None:
        self.connection.execute('INSERT OR REPLACE INTO features VALUES (?,?)',
                                (self.key(text, statute), np.asarray(values, dtype=np.float32).tobytes()))

    def commit(self) -> None:
        self.connection.commit()
