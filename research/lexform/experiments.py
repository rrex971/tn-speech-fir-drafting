from __future__ import annotations

import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import Config, ROOT
from .data import anchor_partition, handwritten, load_statutes, mapping_rows, section_id, validate_splits
from .metrics import calibrate, multilabel, risk_curve, select, selective_point, tune_threshold
from .results import record, require_baseline, run_path, write_json


def seed_all(seed: int) -> None:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def label_ids() -> list[str]:
    values = json.loads((ROOT / 'data/mapping/ilsi_ipc_labels.json').read_text())
    return [section_id(key) for key, _ in sorted(values.items(), key=lambda pair: pair[1])]


def targets(records: list[dict], labels: list[str]) -> np.ndarray:
    lookup = {label: i for i, label in enumerate(labels)}
    result = np.zeros((len(records), len(labels)), dtype=bool)
    for i, row in enumerate(records):
        for label in row['labels']:
            if label in lookup:
                result[i, lookup[label]] = True
    return result


def baseline(directory: Path, cfg: Config, output: Path) -> dict:
    from fir.data.ilsi import FactInstance
    from fir.statute.labels import LabelSpace
    from fir.statute.tfidf_baseline import TfidfStatuteClassifier
    seed_all(cfg.seed)
    manifest, splits = validate_splits(directory, cfg.protocol)
    if not manifest['full_split']:
        raise ValueError(f'P0 requires exact fixed split sizes: {manifest["counts"]}')
    provenance_path = directory / 'provenance.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if cfg.protocol == 'canonical' and provenance and provenance.get('canonical_zenodo_files') is False:
        raise ValueError('Supplemental data requires protocol="supplement". It cannot reproduce a canonical raw-ILSI benchmark.')
    if cfg.protocol == 'supplement' and not (provenance and provenance.get('profile') == 'sigr-ilsi'):
        raise ValueError('Supplement protocol requires a verified import provenance manifest')
    labels = label_ids()
    classifier = TfidfStatuteClassifier(LabelSpace(labels=[f'Section {label} in The Indian Penal Code' for label in labels]))
    train = [FactInstance(id=row['id'], text=row['text'], ipc_labels=[f'Section {label} in The Indian Penal Code' for label in row['labels']])
             for row in splits['train']]
    classifier.fit(train)
    dev = classifier.predict_proba([row['text'] for row in splits['dev']])
    threshold = tune_threshold(targets(splits['dev'], labels), dev)
    scores = classifier.predict_proba([row['text'] for row in splits['test']])
    truth = targets(splits['test'], labels)
    classifier.threshold = threshold
    classifier.save(output / 'tfidf.joblib')
    np.savez_compressed(output / 'predictions.npz', dev_scores=dev, dev_y=targets(splits['dev'], labels),
                        test_scores=scores, test_y=truth)
    metrics = multilabel(truth, scores >= threshold)
    from fir.config import pipeline
    return record(output, 'P0', 'completed', config=cfg.to_dict(), manifest=manifest, metrics=metrics,
                  threshold=threshold, labels=labels, model='tfidf+ovr-logistic',
                  tfidf_settings=pipeline()['statute_id']['tfidf'], class_weight='balanced', solver='liblinear',
                  gate_passed=(abs(metrics['micro_f1'] - 0.700) <= 0.01) if cfg.protocol == 'canonical' else metrics['micro_f1'] > 0,
                  historical_reference_applicable=cfg.protocol == 'canonical', dataset_provenance=provenance,
                  gate_protocol='raw-ILSI historical reproduction' if cfg.protocol == 'canonical' else 'full-split supplemental baseline measured',
                  baseline_reference={'micro_f1': 0.700, 'macro_f1': 0.580, 'source': 'user-supplied historical result'},
                  vectorizer=classifier.vectorizer_mode, train_seconds=classifier.train_seconds)


def _context(directory: Path, cfg: Config):
    try:
        baseline_result = require_baseline(cfg.protocol)
    except ValueError:
        if cfg.protocol != 'supplement':
            raise
        baseline_result = None
    manifest, splits = validate_splits(directory, cfg.protocol)
    if not manifest['full_split']:
        raise ValueError('Source split counts do not match the selected protocol')
    if baseline_result and manifest['sha256'] != baseline_result['manifest']['sha256']:
        raise ValueError('Dataset hashes differ from the passing P0 run')
    statutes = load_statutes(directory)
    lookup = {row['id']: row['text'] for row in statutes if row['code'] == 'IPC'}
    labels = label_ids()
    if cfg.variant == 'handwritten':
        covered = {row['section'] for row in handwritten()}
        labels = [label for label in labels if label in covered]
        splits = {key: [row for row in values if set(row['labels']) & covered] for key, values in splits.items()}
    return manifest, splits, labels, [lookup[label] for label in labels], statutes


def prepare_features(directory: Path, cfg: Config, output: Path) -> dict:
    import torch
    from .model import TextEncoder
    if not cfg.freeze_encoder or cfg.pooling == 'attention':
        raise ValueError('Feature preparation requires a frozen encoder with mean or CLS pooling')
    seed_all(cfg.seed)
    manifest, splits = validate_splits(directory, cfg.protocol)
    if not manifest['full_split']:
        raise ValueError('Source split counts do not match the selected protocol')
    encoder = TextEncoder(cfg).to(cfg.device).eval()
    encoder.enable_feature_cache()
    texts = list(dict.fromkeys(row['text'] for split in splits.values() for row in split))
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=cfg.device, dtype=torch.bfloat16, enabled=cfg.device == 'cuda'):
        for start in range(0, len(texts), cfg.feature_batch_size):
            encoder(texts[start:start + cfg.feature_batch_size])
            if start % (cfg.feature_batch_size * 50) == 0:
                elapsed = time.perf_counter() - started
                print(f'features={min(start + cfg.feature_batch_size, len(texts))}/{len(texts)} elapsed={elapsed:.1f}s', flush=True)
    return record(output, 'features', 'completed', config=cfg.to_dict(), documents=len(texts), manifest=manifest,
                  elapsed_seconds=time.perf_counter() - started, benchmark=False)


def train(directory: Path, cfg: Config, output: Path, *, phase: str = 'P3') -> dict:
    import torch
    from .model import LexForm, TextEncoder, classification_loss, form_alignment
    seed_all(cfg.seed)
    manifest, splits, labels, statute_texts, all_statutes = _context(directory, cfg)
    if phase == 'P3' and cfg.bank_initialization == 'decomposed':
        from .decomposition import cached_forms
        forms = cached_forms(directory, allow_unsupported=cfg.protocol == 'supplement')
    else:
        forms = []
    device = torch.device(cfg.device)
    encoder = TextEncoder(cfg)
    encoder.enable_feature_cache()
    model = LexForm(cfg, len(labels), encoder).to(device)
    if encoder.feature_store is not None:
        all_texts = list(dict.fromkeys(row['text'] for split in splits.values() for row in split))
        print(f'Preparing frozen encoder features for {len(all_texts)} documents', flush=True)
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            for start in range(0, len(all_texts), cfg.feature_batch_size):
                encoder(all_texts[start:start + cfg.feature_batch_size])
                if start % (cfg.feature_batch_size * 50) == 0:
                    print(f'features={min(start + cfg.feature_batch_size, len(all_texts))}/{len(all_texts)}', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    y = targets(splits['train'], labels)
    positive_weight = torch.tensor((len(y) - y.sum(0)) / np.maximum(y.sum(0), 1), dtype=torch.float32, device=device).clamp(max=100)
    human = handwritten()
    grounding_texts = list(dict.fromkeys(row['text'] for row in human))
    induced_texts = list(dict.fromkeys(clause['text'] for form in forms for group in form['groups'] for clause in group['alternatives']))
    anchor_train, anchor_test = anchor_partition(mapping_rows(), cfg.seed, cfg.anchor_fraction)
    source_index = {label: i for i, label in enumerate(labels)}
    anchor_train = [row for row in anchor_train if row['ipc_section'] in source_index]
    bns_texts = {row['id']: row['text'] for row in all_statutes if row['code'] == 'BNS'}
    fixed = None
    model.eval()
    with torch.no_grad():
        descriptions = grounding_texts + induced_texts[:max(0, cfg.bank_size - len(grounding_texts))]
        initial = encoder(descriptions, statute=True)
        count = min(len(initial), cfg.bank_size)
        model.bank[:count].copy_(initial[:count])
        if cfg.variant == 'handwritten':
            if len(grounding_texts) > cfg.bank_size:
                raise ValueError('Handwritten baseline needs a bank large enough for all reference elements')
            fixed = torch.zeros(len(labels), cfg.groups, cfg.bank_size, device=device)
            for i, label in enumerate(labels):
                entries = [row for row in human if row['section'] == label]
                for group, row in enumerate(entries):
                    fixed[i, group, grounding_texts.index(row['text'])] = 1
    history, best = [], -1.0
    start_time = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        order = np.random.default_rng(cfg.seed + epoch).permutation(len(y))
        batches = [order[start:start + cfg.batch_size] for start in range(0, len(order), cfg.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, indices in enumerate(batches):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                output_batch = model([splits['train'][int(i)]['text'] for i in indices], statute_texts, fixed)
                truth = torch.tensor(y[indices], dtype=torch.float32, device=device)
                loss = classification_loss(output_batch['scores'], truth, cfg.loss, positive_weight)
                if cfg.variant != 'flat':
                    element_vectors = encoder(grounding_texts, statute=True)
                    loss = loss + cfg.grounding_weight * model.grounding_loss(element_vectors)
                    probabilities = output_batch['probabilities']
                    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean()
                    loss = loss + cfg.sparsity_weight * entropy
                    if anchor_train and cfg.anchor_weight and cfg.variant not in {'per_section', 'handwritten'}:
                        anchor_sample = random.sample(anchor_train, min(4, len(anchor_train)))
                        target_vectors = encoder([bns_texts[row['target']] for row in anchor_sample], statute=True)
                        _, target_forms = model.induce(target_vectors)
                        source_forms = probabilities[[source_index[row['ipc_section']] for row in anchor_sample]]
                        loss = loss + cfg.anchor_weight * form_alignment(source_forms, target_forms)
            window_size = min(cfg.accumulation, len(batches) - (step // cfg.accumulation) * cfg.accumulation)
            (loss / window_size).backward()
            total_loss += float(loss.detach())
            if (step + 1) % cfg.accumulation == 0 or step == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(f'epoch={epoch + 1} batch={step + 1}/{len(batches)} loss={float(loss.detach()):.5f}', flush=True)
        dev_scores, dev_weakest = predict(model, splits['dev'], statute_texts, cfg, fixed)
        dev_truth = targets(splits['dev'], labels)
        threshold = tune_threshold(dev_truth, dev_scores)
        dev_metrics = multilabel(dev_truth, dev_scores >= threshold)
        history.append({'epoch': epoch + 1, 'loss': total_loss / max(1, len(batches)), 'dev': dev_metrics})
        if dev_metrics['micro_f1'] > best:
            best = dev_metrics['micro_f1']
            encoder.tokenizer.save_pretrained(output / 'encoder')
            encoder.encoder.config.save_pretrained(output / 'encoder')
            state = {name: tensor for name, tensor in model.state_dict().items()
                     if not (cfg.freeze_encoder and name.startswith('text_encoder.encoder.'))}
            torch.save({'state_dict': state, 'fixed_forms': fixed.cpu() if fixed is not None else None}, output / 'model.pt')
        write_json(output / 'history.json', history)
    metadata = {'config': cfg.to_dict(), 'labels': labels, 'statute_texts': statute_texts,
                'anchor_train': [row['ipc_section'] for row in anchor_train],
                'anchor_heldout': [row['ipc_section'] for row in anchor_test],
                'label_scope': 'annotated_29_only' if cfg.variant == 'handwritten' else 'all_100',
                'dataset_sha256': manifest['sha256'], 'training_completed': True, 'research_only': True,
                'bank_descriptions': descriptions[:cfg.bank_size],
                'bank_initialization': cfg.bank_initialization,
                'decomposition_coverage': {'valid': len(forms), 'total': len(all_statutes),
                                         'unsupported_excluded': len(all_statutes) - len(forms)}
                                         if phase == 'P3' and cfg.bank_initialization == 'decomposed' else None,
                'bank_description_status': 'initialization labels; learned prototypes require inspection'}
    write_json(output / 'checkpoint.json', metadata)
    saved = torch.load(output / 'model.pt', map_location=device, weights_only=True)
    _restore_state(model, saved['state_dict'], cfg.freeze_encoder)
    if cfg.freeze_encoder:
        save_statute_features(model, all_statutes, output)
        save_learned_forms(model, all_statutes, descriptions, output)
    dev_scores, dev_weakest = predict(model, splits['dev'], statute_texts, cfg, fixed)
    test_scores, test_weakest = predict(model, splits['test'], statute_texts, cfg, fixed)
    dev_truth, test_truth = targets(splits['dev'], labels), targets(splits['test'], labels)
    threshold = tune_threshold(dev_truth, dev_scores)
    np.savez_compressed(output / 'predictions.npz', dev_scores=dev_scores, dev_weakest=dev_weakest,
                        dev_y=dev_truth, test_scores=test_scores, test_weakest=test_weakest, test_y=test_truth)
    metrics = multilabel(test_truth, test_scores >= threshold)
    return record(output, phase, 'completed', config=cfg.to_dict(), manifest=manifest, metrics=metrics,
                  training_constants={'optimizer': 'AdamW', 'weight_decay': 0.01, 'gradient_clip': 1.0,
                                      'positive_weight_cap': 100, 'cuda_autocast': 'bfloat16',
                                      'form_membership_top_k': 4, 'focal_gamma': 2, 'asymmetric_negative_gamma': 4,
                                      'asymmetric_negative_clip': 0.05},
                  threshold=threshold, history=history, label_scope=metadata['label_scope'],
                  anchor_train=metadata['anchor_train'], anchor_heldout=metadata['anchor_heldout'],
                  gate_passed=(metrics['micro_f1'] > 0.700 and metrics['macro_f1'] > 0.580)
                  if cfg.variant == 'and_or' and cfg.protocol == 'canonical' else None,
                  elapsed_seconds=time.perf_counter() - start_time,
                  interpretation='Mapping-supervised transfer' if cfg.anchor_weight else 'No cross-code anchor supervision')


def save_statute_features(model, statutes: list[dict], output: Path) -> None:
    import hashlib
    import torch
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(statutes), 16):
            values.append(model.text_encoder.pooled([row['text'] for row in statutes[start:start + 16]], True).cpu().numpy())
    np.savez_compressed(output / 'statute_features.npz', features=np.concatenate(values),
                        keys=[f"{row['code']}:{row['id']}" for row in statutes],
                        hashes=[hashlib.sha256(row['text'].encode()).hexdigest() for row in statutes])


def save_learned_forms(model, statutes: list[dict], descriptions: list[str], output: Path) -> None:
    import torch
    from torch.nn.functional import normalize
    if model.cfg.variant in {'flat', 'per_section', 'handwritten'}:
        return
    with np.load(output / 'statute_features.npz', allow_pickle=False) as stored, torch.inference_mode():
        raw = torch.from_numpy(stored['features'].copy()).to(model.bank.device)
        vectors = normalize(model.text_encoder.projection(raw).float(), dim=-1)
        membership, _ = model.induce(vectors)
        result = {}
        for row, form in zip(statutes, membership):
            groups = []
            for group in form:
                alternatives = [{'predicate': index,
                                 'text': descriptions[index] if index < len(descriptions) else f'Learned prototype p{index}',
                                 'membership': float(group[index])} for index in group.argsort(descending=True)[:4].tolist()
                                if float(group[index]) > 0]
                groups.append({'alternatives': alternatives})
            result[f"{row['code']}:{row['id']}"] = {'groups': groups, 'reviewed': False,
                'kind': 'learned_program', 'note': 'Descriptions identify initialization, not verified learned semantics.'}
    write_json(output / 'learned_forms.json', result)


def predict(model, rows: list[dict], statutes: list[str], cfg: Config, fixed=None):
    import torch
    model.eval()
    scores, weakest = [], []
    with torch.inference_mode():
        section_vectors = model.text_encoder(statutes, statute=True) if cfg.variant != 'flat' else None
        for start in range(0, len(rows), cfg.batch_size):
            output = model([row['text'] for row in rows[start:start + cfg.batch_size]], statutes, fixed, section_vectors)
            scores.append(output['scores'].float().cpu().numpy())
            weakest.append(output['groups'].amin(-1).float().cpu().numpy())
    return np.concatenate(scores), np.concatenate(weakest)


def load_checkpoint(directory: Path, device: str = 'cpu'):
    import torch
    from .model import LexForm, TextEncoder
    metadata = json.loads((directory / 'checkpoint.json').read_text())
    if metadata.get('training_completed') is not True:
        raise ValueError('Checkpoint is not a completed training run')
    cfg = Config(**metadata['config'])
    cfg.device = device
    model = LexForm(cfg, len(metadata['labels']), TextEncoder(cfg, directory / 'encoder')).to(device)
    saved = torch.load(directory / 'model.pt', map_location=device, weights_only=True)
    _restore_state(model, saved['state_dict'], cfg.freeze_encoder)
    fixed = saved['fixed_forms']
    return model.eval(), cfg, metadata, fixed.to(device) if fixed is not None else None


def _restore_state(model, state: dict, frozen: bool) -> None:
    if not frozen:
        model.load_state_dict(state)
        return
    keys = model.load_state_dict(state, strict=False)
    if keys.unexpected_keys or any(not key.startswith('text_encoder.encoder.') for key in keys.missing_keys):
        raise ValueError('Checkpoint is missing trained layers or has unexpected tensors')


def calibration(checkpoint: Path, output: Path, cfg: Config) -> dict:
    metadata = json.loads((checkpoint / 'checkpoint.json').read_text())
    try:
        baseline_result = require_baseline(metadata['config'].get('protocol', 'canonical'))
    except ValueError:
        if metadata['config'].get('protocol') != 'supplement':
            raise
        baseline_result = None
    if baseline_result and metadata['dataset_sha256'] != baseline_result['manifest']['sha256']:
        raise ValueError('P0 and trained checkpoint use different splits')
    values = np.load(checkpoint / 'predictions.npz')
    chosen = calibrate(values['dev_y'], values['dev_scores'], values['dev_weakest'], cfg.target_risk)
    prediction = select(values['test_scores'], values['test_weakest'], chosen['threshold'], chosen['floor'])
    curve = risk_curve(values['test_y'], values['test_scores'], values['test_weakest'], chosen['floor'])
    from .config import output_dir
    if metadata['label_scope'] != 'all_100':
        raise ValueError('Matched-coverage comparison requires identical all-100 label scope')
    baseline_curve = []
    if baseline_result:
        baseline_path = output_dir() / 'runs' / baseline_result['id'] / 'predictions.npz'
        with np.load(baseline_path) as base:
            baseline_curve = risk_curve(base['test_y'], base['test_scores'], base['test_scores'], 0)
    matched = []
    for point in curve if baseline_curve else []:
        peer = min(baseline_curve, key=lambda candidate: abs(candidate['coverage'] - point['coverage']))
        matched.append({'coverage': point['coverage'], 'risk': point['risk'],
                        'baseline_coverage': peer['coverage'], 'baseline_risk': peer['risk']})
    report = {'checkpoint_id': checkpoint.name, 'config': cfg.to_dict(), 'calibration': chosen,
              'metrics': multilabel(values['test_y'], prediction), 'selective': selective_point(values['test_y'], prediction),
              'risk_coverage': curve, 'baseline_risk_coverage': baseline_curve, 'matched_coverage': matched,
              'baseline_comparison_available': baseline_result is not None,
              'coverage_matching': 'nearest measured test coverage; residual reported, no test-set tuning'}
    result = record(output, 'P4', 'completed', **report)
    write_json(checkpoint / 'calibration.json', chosen)
    return result


def transfer(directory: Path, checkpoint: Path, output: Path) -> dict:
    from .config import output_dir
    model, cfg, metadata, fixed = load_checkpoint(checkpoint)
    require_baseline(cfg.protocol)
    if cfg.variant in {'flat', 'per_section', 'handwritten'}:
        raise ValueError('This ablation cannot induce forms for unseen target sections')
    manifest, splits = validate_splits(directory, cfg.protocol)
    if manifest['sha256'] != metadata['dataset_sha256']:
        raise ValueError('Transfer split hashes differ from training')
    calibration_path = checkpoint / 'calibration.json'
    if not calibration_path.exists():
        raise ValueError('Run P4 calibration on the IPC development split before P5')
    calibration_values = json.loads(calibration_path.read_text())
    all_bns = [row for row in load_statutes(directory) if row['code'] == 'BNS']
    scores, weakest = predict(model, splits['test'], [row['text'] for row in all_bns], cfg)
    prediction = select(scores, weakest, calibration_values['threshold'], calibration_values['floor'])
    rows = mapping_rows()
    heldout = set(metadata['anchor_heldout'])
    eligible = {row['ipc_section']: row['target'] for row in rows if row['eligible']}
    target_ids = [row['id'] for row in all_bns]
    target_lookup = {section: i for i, section in enumerate(target_ids)}
    gold = np.zeros_like(prediction)
    included, heldout_docs, unsupported_only = [], [], []
    non_bns = {row['ipc_section'] for row in rows if row['target'] is None}
    for i, complaint in enumerate(splits['test']):
        source = set(complaint['labels'])
        if source and source <= set(eligible):
            included.append(i)
            for label in source:
                gold[i, target_lookup[eligible[label]]] = True
            if source <= heldout:
                heldout_docs.append(i)
        if source and source <= non_bns:
            unsupported_only.append(i)
    evaluation_columns = sorted({target_lookup[value] for value in eligible.values()})
    heldout_columns = sorted({target_lookup[eligible[label]] for label in heldout if label in eligible})
    def subset_metrics(documents, columns):
        return multilabel(gold[np.ix_(documents, columns)], prediction[np.ix_(documents, columns)]) if documents and columns else None
    result = record(output, 'P5', 'completed', checkpoint_id=checkpoint.name, config=cfg.to_dict(),
                    protocol='mapping_proxy_not_independently_annotated_BNS_ground_truth',
                    all_eligible_proxy=subset_metrics(included, evaluation_columns),
                    heldout_anchor_proxy=subset_metrics(heldout_docs, heldout_columns),
                    heldout_documents=len(heldout_docs), unsupported_only_documents=len(unsupported_only),
                    non_bns_source_sections=sorted(non_bns),
                    unsupported_abstention_rate=float((~prediction[unsupported_only].any(1)).mean()) if unsupported_only else None,
                    scored_target_sections=len(evaluation_columns), target_sections_total=len(target_ids),
                    unscored_target_predictions=int(prediction[:, [i for i in range(len(target_ids)) if i not in evaluation_columns]].sum()),
                    limitations=['Correspondence is proxy supervision, not a new-code complaint annotation.',
                                 'Mixed supported/unsupported source labels are excluded from the abstention metric.',
                                 'All-358 BNS prediction quality requires independently annotated BNS cases.',
                                 'Anchored training is not strict unsupervised zero-shot transfer.'])
    return result


def diagnosis(directory: Path, cfg: Config, output: Path) -> dict:
    baseline_cfg = replace(cfg, variant='flat', loss='bce', pooling='cls', truncation='512', anchor_weight=0)
    variants = [('control', baseline_cfg)]
    variants += [(f'loss-{loss}', replace(baseline_cfg, loss=loss)) for loss in ('pos_weight', 'focal', 'asymmetric')]
    variants += [('chunked', replace(baseline_cfg, truncation='chunked'))]
    variants += [(f'pool-{pooling}', replace(baseline_cfg, pooling=pooling)) for pooling in ('mean', 'attention')]
    measurements = []
    for name, experiment_cfg in variants:
        path = run_path(f'{output.name}-{name}')
        result = train(directory, experiment_cfg, path, phase='P1-experiment')
        measurements.append({'id': result['id'], 'factor': name, 'metrics': result['metrics']})
        write_json(output / 'progress.json', measurements)
    return record(output, 'P1', 'completed', config=cfg.to_dict(), experiments=measurements,
                  dataset_sha256=result['manifest']['sha256'],
                  design='one factor at a time; identical seed, splits and flat 100-label head',
                  interpretation='Compare imbalance, context length and pooling; no causal explanation is assumed in advance.')
