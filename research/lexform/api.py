from __future__ import annotations

import json
import hashlib
import re
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .config import data_dir, output_dir
from .data import audit, handwritten, jsonl
from .config import ROOT
from .results import runs

router = APIRouter(prefix='/v1/research/lexform', tags=['LexForm research'])
_model_lock = threading.Lock()


def bundled_runs() -> list[dict]:
    result = []
    root = ROOT / 'research/lexform/bundled'
    paths = [*root.glob('*/result.json'), *root.glob('*/evidence/*/result.json')]
    for path in sorted(paths):
        try:
            item = json.loads(path.read_text())
            if item['id'] == path.parent.name:
                result.append(item)
        except (OSError, ValueError, KeyError):
            continue
    return result


def measurements() -> list[dict]:
    unique = {item['id']: item for item in bundled_runs() + runs()}
    return sorted(unique.values(), key=lambda item: item.get('created_at', ''), reverse=True)


def checkpoint_path(checkpoint_id: str) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', checkpoint_id):
        raise ValueError('Invalid checkpoint ID')
    for root in (output_dir() / 'runs', ROOT / 'research/lexform/bundled'):
        path = root / checkpoint_id
        if (path / 'checkpoint.json').is_file() and (path / 'model.pt').is_file():
            return path
    raise FileNotFoundError('Checkpoint not found')


def completed_checkpoints() -> list[dict]:
    result = []
    for item in measurements():
        if item['status'] == 'completed' and item['phase'] == 'P3':
            try:
                path = checkpoint_path(item['id'])
                if (path / 'calibration.json').is_file():
                    result.append(item)
            except FileNotFoundError:
                continue
    return result


@router.get('/status')
def status() -> dict:
    measured = measurements()
    inputs = audit(data_dir())
    phases = []
    titles = ['Baseline harness', 'Encoder diagnosis', 'Element decomposition', 'LexForm training',
              'Selective prediction', 'IPC → BNS transfer', 'Results report']
    for number, title in enumerate(titles):
        matching = [item for item in measured if item['phase'] == f'P{number}']
        completed = [item for item in matching if item['status'] == 'completed']
        gate = any(item.get('gate_passed') is True for item in completed)
        phase_status = ('passed' if gate else 'measured') if completed else matching[0]['status'] if matching else 'pending'
        if number == 6 and completed and not completed[0].get('all_experiments_completed'):
            phase_status = 'partial'
        phases.append({'id': f'P{number}', 'title': title, 'status': phase_status,
                       'run_id': completed[0]['id'] if completed else None})
    return {'name': 'LexForm', 'research_only': True, 'audit': inputs, 'phases': phases,
            'checkpoints': [{'id': item['id'], 'variant': item['config']['variant']} for item in completed_checkpoints()],
            'completed_experiments': sum(item['status'] == 'completed' and item['phase'] in {'P0', 'P1-experiment', 'P3', 'P4', 'P5'} for item in measured),
            'reference_target': {'micro_f1': 0.700, 'macro_f1': 0.580, 'measured_locally': False}}


@router.get('/runs')
def list_runs() -> list[dict]:
    return [{key: value for key, value in item.items() if key not in {'manifest', 'environment', 'command'}} for item in measurements()]


@router.get('/statutes')
def statutes() -> list[dict]:
    references = handwritten()
    result = []
    for section in sorted({row['section'] for row in references}, key=lambda value: int(re.match(r'\d+', value).group())):
        result.append({'id': section, 'code': 'IPC', 'heading': '', 'text': None,
                       'reference_elements': [row for row in references if row['section'] == section]})
    for row in jsonl(ROOT / 'data/statutes/bns_2023.jsonl'):
        result.append({'id': row['section'], 'code': 'BNS', 'heading': row['heading'], 'text': row['text'], 'reference_elements': []})
    available = completed_checkpoints()
    learned = {}
    if available:
        metadata = json.loads((checkpoint_path(available[0]['id']) / 'checkpoint.json').read_text())
        learned_path = checkpoint_path(available[0]['id']) / 'learned_forms.json'
        if learned_path.is_file():
            learned = json.loads(learned_path.read_text())
        ipc_texts = dict(zip(metadata['labels'], metadata['statute_texts']))
        for row in result:
            if row['code'] == 'IPC':
                row['text'] = ipc_texts.get(row['id'])
        existing = {row['id'] for row in result if row['code'] == 'IPC'}
        result.extend({'id': section, 'code': 'IPC', 'heading': '', 'text': text, 'reference_elements': []}
                      for section, text in ipc_texts.items() if section not in existing)
    for row in result:
        path = output_dir() / 'decompositions' / f"{row['code']}-{row['id']}.json"
        if not path.exists() and available:
            path = checkpoint_path(available[0]['id']) / 'forms' / path.name
        try:
            payload = json.loads(path.read_text()) if path.exists() else None
            if payload and payload.get('unsupported'):
                row['decomposition_note'] = payload.get('reason')
                payload = None
            if payload and row['text'] and payload.get('source_sha256') != hashlib.sha256(row['text'].encode()).hexdigest():
                payload = None
            row['induced_form'] = payload
        except (OSError, ValueError):
            row['induced_form'] = None
        row['learned_form'] = learned.get(f"{row['code']}:{row['id']}")
    return result


class Inspection(BaseModel):
    text: str = Field(min_length=5, max_length=20000)
    checkpoint_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,100}$')
    code: str = Field(default='BNS', pattern='^(IPC|BNS)$')


@lru_cache(maxsize=1)
def resident_model(checkpoint_id: str):
    from .experiments import load_checkpoint
    path = checkpoint_path(checkpoint_id)
    manifest = path / 'bundle.json'
    if manifest.is_file():
        entries = json.loads(manifest.read_text())['files']
        for name, expected in entries.items():
            file = (path / name).resolve()
            if not file.is_relative_to(path.resolve()) or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError('Bundled checkpoint integrity check failed')
    return load_checkpoint(path)


@lru_cache(maxsize=2)
def statute_vectors(checkpoint_id: str, code: str):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    model, cfg, metadata, _ = resident_model(checkpoint_id)
    if code == 'IPC':
        ids, texts = metadata['labels'], metadata['statute_texts']
    else:
        sections = list(jsonl(ROOT / 'data/statutes/bns_2023.jsonl'))
        ids, texts = [str(row['section']) for row in sections], [row['text'] for row in sections]
    with torch.inference_mode():
        path = checkpoint_path(checkpoint_id) / 'statute_features.npz'
        if cfg.freeze_encoder and path.is_file():
            with np.load(path, allow_pickle=False) as stored:
                lookup = {str(key): i for i, key in enumerate(stored['keys'])}
                indices = [lookup[f'{code}:{section}'] for section in ids]
                if any(stored['hashes'][i] != hashlib.sha256(text.encode()).hexdigest() for i, text in zip(indices, texts)):
                    raise ValueError('Cached statutory text hashes do not match')
                raw = torch.from_numpy(stored['features'][indices].copy())
            vectors = functional.normalize(model.text_encoder.projection(raw).float(), dim=-1)
        else:
            vectors = model.text_encoder(texts, statute=True)
    return ids, texts, vectors


def warm_checkpoint() -> None:
    available = completed_checkpoints()
    if available:
        with _model_lock:
            checkpoint = available[0]['id']
            model, _, _, _ = resident_model(checkpoint)
            import torch
            with torch.inference_mode():
                model.text_encoder(['Model warm-up.'])
            statute_vectors(checkpoint, 'IPC')
            if available[0]['config']['variant'] not in {'flat', 'per_section', 'handwritten'}:
                statute_vectors(checkpoint, 'BNS')


@router.post('/inspect')
def inspect(body: Inspection) -> dict:
    available = {item['id'] for item in completed_checkpoints()}
    if body.checkpoint_id not in available:
        raise HTTPException(409, 'No completed LexForm checkpoint with that ID. Run the gated training phases first.')
    try:
        import torch
        with _model_lock:
            model, cfg, metadata, fixed = resident_model(body.checkpoint_id)
            if body.code == 'BNS' and cfg.variant in {'flat', 'per_section', 'handwritten'}:
                raise HTTPException(409, 'This ablation has no unseen-section form inducer.')
            ids, texts, vectors = statute_vectors(body.checkpoint_id, body.code)
            calibration_file = checkpoint_path(body.checkpoint_id) / 'calibration.json'
            if not calibration_file.exists():
                raise HTTPException(409, 'Calibrate this checkpoint in P4 before inspection.')
            calibration = json.loads(calibration_file.read_text())
            with torch.inference_mode():
                output = model([body.text.strip()], texts, fixed if body.code == 'IPC' else None, vectors)
            scores = output['scores'][0].cpu()
            weakest = output['groups'][0].amin(-1).cpu()
            predictions = []
            seen = set()
            for index in scores.argsort(descending=True).tolist():
                if ids[index] in seen:
                    continue
                seen.add(ids[index])
                score, floor = float(scores[index]), float(weakest[index])
                groups = []
                if 'membership' in output:
                    for group_index, membership in enumerate(output['membership'][index]):
                        alternatives = []
                        for bank_index in membership.argsort(descending=True)[:4].tolist():
                            if float(membership[bank_index]) <= 0:
                                continue
                            descriptions = metadata.get('bank_descriptions', [])
                            alternatives.append({'predicate': bank_index,
                                                 'initialization_label': descriptions[bank_index] if bank_index < len(descriptions) else None,
                                                 'membership': float(membership[bank_index])})
                        groups.append({'score': float(output['groups'][0, index, group_index]), 'alternatives': alternatives})
                heading = next((row.get('heading', '') for row in jsonl(ROOT / 'data/statutes/bns_2023.jsonl')
                                if str(row['section']) == ids[index]), '') if body.code == 'BNS' else ''
                predictions.append({'section': ids[index], 'heading': heading, 'score': score, 'weakest_group': floor,
                                    'accepted': score >= calibration['threshold'] and floor >= calibration['floor'],
                                    'groups': groups})
                if len(predictions) == 8:
                    break
            return {'code': body.code, 'checkpoint_id': body.checkpoint_id, 'predictions': predictions,
                    'calibration': calibration, 'research_only': True, 'changes_fir_draft': False}
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(503, f'LexForm checkpoint unavailable: {exc}') from exc
