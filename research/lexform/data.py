from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .config import ROOT

EXPECTED = {'train': 42835, 'dev': 10200, 'test': 13039}
SUPPLEMENT_EXPECTED = {'train': 42835, 'dev': 10200, 'test': 13039}


def section_id(value: object) -> str:
    text = str(value).strip()
    match = re.search(r'(?:Section\s+|IPC\s+)?(\d+[A-Za-z]*(?:\([A-Za-z0-9]+\))*)', text, re.I)
    if not match:
        raise ValueError(f'Unrecognised section label: {text[:100]}')
    return match.group(1).upper()


def jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing local input: {path}. No data will be downloaded automatically.')
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path.name}:{number}: invalid JSON') from exc


def load_split(directory: Path, split: str, excluded: list | None = None) -> list[dict]:
    if split not in EXPECTED:
        raise ValueError('Expected train, dev or test')
    records = []
    for row in jsonl(directory / f'{split}.jsonl'):
        text = ' '.join(row['text']) if isinstance(row['text'], list) else str(row['text'])
        labels = sorted({section_id(label) for label in row['labels']})
        if not text.strip() or not labels:
            if excluded is not None:
                excluded.append({'id': str(row.get('id')), 'reason': 'empty_text' if not text.strip() else 'empty_labels'})
                continue
            raise ValueError(f'Empty text or labels in {split}: {row.get("id")}')
        records.append({'id': str(row['id']), 'text': text, 'labels': labels})
    return records


def load_statutes(directory: Path) -> list[dict]:
    ipc = []
    for row in jsonl(directory / 'secs.jsonl'):
        text = ' '.join(row['text']) if isinstance(row['text'], list) else str(row['text'])
        ipc.append({'id': section_id(row['id']), 'code': 'IPC', 'text': text, 'heading': ''})
    bns = [{'id': str(row['section']), 'code': 'BNS', 'text': row['text'], 'heading': row['heading']}
           for row in jsonl(ROOT / 'data/statutes/bns_2023.jsonl')]
    return ipc + bns


def mapping_rows() -> list[dict]:
    with (ROOT / 'data/mapping/ipc_bns_map.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        match = re.fullmatch(r'(?:BNS\s+)?(\d+[A-Za-z]*)(?:\([^)]*\))*(?:,\s*\([^)]*\))*', row['bns_section'].strip())
        row['target'] = match.group(1) if match else None
        row['eligible'] = bool(row['target'] and row['confidence'].lower() == 'high'
                               and row['needs_review'].strip().upper() not in {'Y', 'YES', 'TRUE', '1'})
    return rows


def anchor_partition(rows: list[dict], seed: int, fraction: float) -> tuple[list[dict], list[dict]]:
    eligible = [row for row in rows if row['eligible']]
    targets = sorted({row['target'] for row in eligible}, key=lambda target: hashlib.sha256(f'{seed}:{target}'.encode()).hexdigest())
    count = min(int(len(targets) * fraction), max(0, len(targets) - 1))
    train_targets = set(targets[:count])
    return ([row for row in eligible if row['target'] in train_targets],
            [row for row in eligible if row['target'] not in train_targets])


def handwritten() -> list[dict]:
    from fir.statute.elements import ELEMENTS
    return [{'section': section, 'text': element.name, 'note': element.note}
            for section, elements in ELEMENTS.items() for element in elements]


def audit(directory: Path) -> dict:
    elements = handwritten()
    names = [row['text'] for row in elements]
    rows = mapping_rows()
    files = {name: (directory / f'{name}.jsonl').is_file() for name in (*EXPECTED, 'secs')}
    literal_or = sum(bool(re.search(r'\bor\b', name, re.I)) for name in names)
    return {
        'dataset_files': files,
        'data_ready': all(files.values()),
        'expected_counts': EXPECTED,
        'handwritten_sections': len({row['section'] for row in elements}),
        'handwritten_elements': len(elements),
        'literal_or_elements': literal_or,
        'or_or_slash_elements': sum(bool(re.search(r'\bor\b|/', name, re.I)) for name in names),
        'leading_predicates': dict(Counter(name.split()[0].lower() for name in names)),
        'mapping_rows': len(rows),
        'eligible_anchors': sum(row['eligible'] for row in rows),
        'non_bns_rows': [row['ipc_section'] for row in rows if row['target'] is None],
        'bns_sections': sum(1 for _ in jsonl(ROOT / 'data/statutes/bns_2023.jsonl')),
        'claimed_disjunction_count': 29,
        'claim_matches_literal_or': literal_or == 29,
    }


def validate_splits(directory: Path, protocol: str = 'canonical') -> tuple[dict, dict[str, list[dict]]]:
    excluded = {name: [] for name in EXPECTED}
    splits = {name: load_split(directory, name, excluded[name] if protocol == 'supplement' else None) for name in EXPECTED}
    hashes: dict[str, set] = {}
    ids: dict[str, set] = {}
    labels = {section_id(label) for label in json.loads((ROOT / 'data/mapping/ilsi_ipc_labels.json').read_text())}
    report: dict = {'counts': {}, 'sha256': {}, 'overlaps': {}, 'protocol': protocol}
    for name, records in splits.items():
        report['counts'][name] = len(records)
        report['sha256'][name] = hashlib.sha256((directory / f'{name}.jsonl').read_bytes()).hexdigest()
        ids[name] = {row['id'] for row in records}
        hashes[name] = {hashlib.sha256(' '.join(row['text'].split()).encode()).hexdigest() for row in records}
        if len(ids[name]) != len(records):
            raise ValueError(f'Duplicate IDs inside {name}')
        unknown = {label for row in records for label in row['labels']} - labels
        if unknown:
            raise ValueError(f'Unknown labels in {name}: {sorted(unknown)}')
    for left, right in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        overlap = len(hashes[left] & hashes[right])
        report['overlaps'][f'{left}/{right}'] = {'text': overlap, 'ids': len(ids[left] & ids[right])}
        if overlap and protocol == 'canonical':
            raise ValueError(f'Duplicate complaint text across {left}/{right}: {overlap}')
    report['excluded_invalid_documents'] = excluded
    report['original_counts'] = {name: count + len(excluded[name]) for name, count in report['counts'].items()}
    if protocol == 'supplement':
        seen = set(hashes['train'])
        report['excluded_overlap_documents'] = {}
        for name in ('dev', 'test'):
            clean = []
            for row in splits[name]:
                key = hashlib.sha256(' '.join(row['text'].split()).encode()).hexdigest()
                if key not in seen:
                    clean.append(row)
                    seen.add(key)
            report['excluded_overlap_documents'][name] = len(splits[name]) - len(clean)
            splits[name] = clean
            report['counts'][name] = len(clean)
    expected = SUPPLEMENT_EXPECTED if protocol == 'supplement' else EXPECTED
    report['expected_original_counts'] = expected
    report['full_split'] = report['original_counts'] == expected
    return report, splits
