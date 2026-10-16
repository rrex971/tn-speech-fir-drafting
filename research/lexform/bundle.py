from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .config import ROOT, output_dir
from .results import runs, write_json


def bundle(checkpoint: Path) -> dict:
    metadata = json.loads((checkpoint / 'checkpoint.json').read_text())
    result = json.loads((checkpoint / 'result.json').read_text())
    if not metadata.get('training_completed') or result.get('phase') != 'P3' or result.get('status') != 'completed':
        raise ValueError('Only a completed real P3 checkpoint can be bundled')
    if not metadata['config'].get('freeze_encoder'):
        raise ValueError('Portable bundles require a frozen public encoder, with only learned heads saved')
    required = ['model.pt', 'checkpoint.json', 'calibration.json', 'statute_features.npz', 'result.json', 'history.json']
    for name in required:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f'Missing checkpoint artifact: {name}')
    if (checkpoint / 'model.pt').stat().st_size > 40 * 1024 * 1024:
        raise ValueError('Refusing to bundle oversized weights; the pretrained base must not be included')
    target = ROOT / 'research' / 'lexform' / 'bundled' / result['id']
    target.mkdir(parents=True, exist_ok=False)
    for name in required:
        shutil.copy2(checkpoint / name, target / name)
    if (checkpoint / 'learned_forms.json').is_file():
        shutil.copy2(checkpoint / 'learned_forms.json', target / 'learned_forms.json')
    shutil.copytree(checkpoint / 'encoder', target / 'encoder')
    forms = output_dir() / 'decompositions'
    if forms.is_dir():
        shutil.copytree(forms, target / 'forms')
    for item in runs():
        same_data = item.get('manifest', {}).get('sha256') == metadata['dataset_sha256']
        relevant = ((item['phase'] == 'P0' and same_data) or
                    item.get('checkpoint_id') == result['id'] or item['phase'] == 'P2')
        if relevant and item['status'] == 'completed':
            write_json(target / 'evidence' / item['id'] / 'result.json', item)
    manifest = {'id': result['id'], 'source': 'completed local training run', 'contains_training_examples': False,
                'base_encoder': metadata['config']['encoder'], 'revision': metadata['config']['encoder_revision'],
                'files': {str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in target.rglob('*') if path.is_file()}}
    write_json(target / 'bundle.json', manifest)
    return manifest
