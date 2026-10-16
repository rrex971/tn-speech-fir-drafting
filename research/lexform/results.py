from __future__ import annotations

import json
import platform
import re
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import output_dir


def run_path(name: str | None = None) -> Path:
    name = name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', name):
        raise ValueError('Run ID may contain letters, digits, underscores and hyphens')
    target = output_dir() / 'runs' / name
    target.mkdir(parents=True, exist_ok=False)
    return target


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def record(path: Path, phase: str, status: str, **values) -> dict:
    versions = {}
    for package in ('torch', 'transformers', 'scikit-learn', 'numpy'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    result = {'id': path.name, 'phase': phase, 'status': status,
              'created_at': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, '-m', 'research.lexform', *sys.argv[1:]],
              'environment': {'python': platform.python_version(), 'versions': versions}, **values}
    write_json(path / 'result.json', result)
    return result


def runs() -> list[dict]:
    result = []
    for path in sorted((output_dir() / 'runs').glob('*/result.json'), reverse=True):
        try:
            result.append(json.loads(path.read_text()))
        except (ValueError, OSError):
            continue
    return result


def require_baseline(protocol: str = 'canonical') -> dict:
    for result in runs():
        if result['phase'] == 'P0' and result.get('gate_passed') is True and result.get('config', {}).get('protocol', 'canonical') == protocol:
            return result
    raise ValueError('P0 gate is closed: reproduce TF-IDF on the full fixed splits before running this phase.')
