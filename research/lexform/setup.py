from __future__ import annotations

import hashlib
import os
import time
import urllib.request
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import output_dir
from .results import write_json

ENCODER_REVISION = 'b5ecfed8ed6cf9d25a3cb8225a8c52f161f7401a'
DECOMPOSITION_REVISION = 'bb5d59e06d9551d752d08b292a50eb208b07ab1f'

ILSI_FILES = {
    'train.jsonl': 'a1994d1aaa8ae7a987fcb1c2e2dc52f1',
    'dev.jsonl': '237b1776f5a535eb3f079bfd0ea0b200',
    'test.jsonl': '09a554405680830c6d3818320eb055af',
    'secs.jsonl': 'b2768e06377ab032df2d8f8c323886ff',
    'label_vocab.json': '5758e3117c37788045308e03a531a03d',
}


def checksum(path: Path, algorithm: str = 'sha256') -> str:
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, expected_md5: str) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if checksum(target, 'md5') == expected_md5:
            return {'file': target.name, 'verified': True, 'sha256': checksum(target)}
        raise ValueError(f'{target.name} already exists but does not match the published ILSI checksum; preserving it.')
    partial = target.with_suffix(target.suffix + '.part')
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
                mode = 'ab' if offset and response.status == 206 else 'wb'
                with partial.open(mode) as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        stream.write(block)
            if checksum(partial, 'md5') != expected_md5:
                raise ValueError(f'{target.name}: downloaded file failed the published checksum')
            partial.replace(target)
            print(f'{target.name}: verified', flush=True)
            return {'file': target.name, 'verified': True, 'sha256': checksum(target)}
        except (OSError, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f'{target.name}: {exc}') from exc
            time.sleep(2)
    raise RuntimeError('Download did not finish')


def prepare(directory: Path, assets: str = 'all') -> dict:
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    result = {'initial_setup': True, 'runtime_downloads': False, 'assets': {}, 'errors': {}}
    if assets in {'all', 'dataset'}:
        def fetch(name):
            return download(f'https://zenodo.org/records/6053791/files/{name}?download=1', directory / name, ILSI_FILES[name])
        with ThreadPoolExecutor(max_workers=3) as workers:
            futures = {name: workers.submit(fetch, name) for name in ILSI_FILES}
            for name, future in futures.items():
                try:
                    result['assets'][name] = future.result()
                except Exception as exc:
                    result['errors'][name] = str(exc)
    if assets in {'all', 'encoder', 'decomposition'}:
        from huggingface_hub import snapshot_download
        if assets in {'all', 'encoder'}:
            try:
                path = snapshot_download('law-ai/InLegalBERT', revision=ENCODER_REVISION, allow_patterns=[
                    'config.json', 'pytorch_model.bin', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt'],
                    token=False, max_workers=2)
                result['assets']['encoder'] = {'repository': 'law-ai/InLegalBERT', 'revision': Path(path).name,
                                               'path': path, 'verified': True}
            except Exception as exc:
                result['errors']['encoder'] = str(exc)
        if assets in {'all', 'decomposition'}:
            try:
                path = Path(snapshot_download('Qwen/Qwen2.5-7B-Instruct-GGUF', revision=DECOMPOSITION_REVISION,
                                              allow_patterns=['*q4_k_m*.gguf'],
                                              token=False, max_workers=2))
                shards = sorted(path.glob('*q4_k_m*.gguf'))
                if not shards:
                    raise FileNotFoundError('No Q4_K_M shards found after download')
                result['assets']['decomposition'] = {'repository': 'Qwen/Qwen2.5-7B-Instruct-GGUF',
                    'revision': path.name, 'path': str(shards[0]), 'verified': True,
                    'files': [{'name': shard.name, 'sha256': checksum(shard)} for shard in shards]}
            except Exception as exc:
                result['errors']['decomposition'] = str(exc)
    manifest_path = output_dir() / 'assets.json'
    if manifest_path.exists():
        import json
        previous = json.loads(manifest_path.read_text())
        result['assets'] = {**previous.get('assets', {}), **result['assets']}
    write_json(manifest_path, result)
    return result


def _array_records(path: Path):
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8') as stream:
        buffer = ''
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                raise ValueError('Missing top-level data array in supplemental JSON')
            buffer += block
            match = re.search(r'"data"\s*:\s*\[', buffer)
            if match:
                buffer = buffer[match.end():]
                break
            if len(buffer) > 4 * 1024 * 1024:
                raise ValueError('Expected a top-level data array in the supplemental JSON')
        while True:
            buffer = buffer.lstrip(' \r\n\t,')
            if buffer.startswith(']'):
                return
            try:
                row, end = decoder.raw_decode(buffer)
                buffer = buffer[end:]
                yield row
            except json.JSONDecodeError:
                block = stream.read(1024 * 1024)
                if not block:
                    raise ValueError('Incomplete supplemental data array') from None
                buffer += block


def import_supplement(directory: Path) -> dict:
    source_directory = directory / 'supplement-source'
    source_directory.mkdir(parents=True, exist_ok=True)
    files = {
        'train': ('abhishekkk2002/sigr-law', 'supplementary/ILSI/train2.json', 349769192),
        'dev': ('abhishekkk2002/sigr-law', 'supplementary/ILSI/dev2.json', 92378697),
        'test': ('abhishekkk2002/sigr-law', 'supplementary/ILSI/test2.json', 115031270),
        'secs': ('abhishekkk2002/ilsi-law', 'secs.json', 55258),
    }
    def fetch(item):
        name, (dataset, filename, expected_size) = item
        source = source_directory / f'{name}.json'
        if source.exists() and source.stat().st_size == expected_size:
            return name, source
        from urllib.parse import quote
        url = f'https://www.kaggle.com/api/v1/datasets/download/{dataset}/{quote(filename, safe="")}'
        partial = source.with_suffix('.part')
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        with urllib.request.urlopen(request, timeout=90) as response, partial.open('ab' if offset and response.status == 206 else 'wb') as stream:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(block)
        if partial.stat().st_size != expected_size:
            raise ValueError(f'{name}: expected {expected_size} bytes; received {partial.stat().st_size}')
        partial.replace(source)
        print(f'{name}: downloaded research supplement', flush=True)
        return name, source
    with ThreadPoolExecutor(max_workers=3) as pool:
        downloaded = dict(pool.map(fetch, files.items()))
    provenance = {'profile': 'sigr-ilsi', 'canonical_zenodo_files': False,
                  'source': 'Public SIGR-Law research supplement on Kaggle',
                  'note': 'Anonymized derivative. Historical raw-ILSI scores are not directly comparable.', 'files': {}}
    for name, source in downloaded.items():
        target = directory / f'{name}.jsonl'
        if target.exists():
            raise ValueError(f'Preserving existing input {target}; select an empty directory for supplement import')
        payload = json.loads(source.read_text()) if name == 'secs' else None
        if payload is not None:
            records = payload.get('data', payload) if isinstance(payload, dict) else payload
            if isinstance(records, dict):
                records = [{'id': key, 'text': value} for key, value in records.items()]
        else:
            records = _array_records(source)
        count = 0
        partial = target.with_suffix('.jsonl.part')
        with partial.open('w', encoding='utf-8') as stream:
            for row in records:
                clean = {key: row[key] for key in ('id', 'text', 'labels') if key in row}
                if 'id' not in clean or 'text' not in clean or (name != 'secs' and 'labels' not in clean):
                    raise ValueError(f'{name}: incompatible supplemental schema: {list(row)}')
                stream.write(json.dumps(clean, ensure_ascii=False) + '\n')
                count += 1
        partial.replace(target)
        provenance['files'][name] = {'count': count, 'sha256': checksum(target), 'source_sha256': checksum(source),
                                     'source_dataset': files[name][0], 'source_file': files[name][1]}
        print(f'{name}: imported {count} records', flush=True)
    write_json(directory / 'provenance.json', provenance)
    return provenance
