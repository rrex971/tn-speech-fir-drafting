from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import data_dir, load_config, output_dir
from .data import audit
from .results import record, require_baseline, run_path, runs


def smoke(output: Path) -> dict:
    import torch
    from torch import nn
    from .config import Config
    from .experiments import seed_all
    from .model import LexForm, classification_loss
    seed_all(42)
    torch.set_num_threads(2)

    class SyntheticEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 16)

        def forward(self, texts, statute=False):
            return self.embedding(torch.tensor([int(text) for text in texts]))

    cfg = Config(bank_size=8, dimension=16, groups=2)
    model = LexForm(cfg, 2, SyntheticEncoder())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    y = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
    losses = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        scores = model(['0', '1', '2', '3'], ['4', '5'])['scores']
        loss = classification_loss(scores, y, 'bce')
        loss.backward()
        if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
            raise ValueError('Non-finite gradient in smoke check')
        optimizer.step()
        losses.append(float(loss.detach()))
    if losses[-1] >= losses[0]:
        raise ValueError('Synthetic optimization did not lower loss')
    return record(output, 'smoke', 'completed', config=cfg.to_dict(), benchmark=False,
                  data='four synthetic numeric inputs; tiny embedding encoder, not InLegalBERT',
                  initial_loss=losses[0], final_loss=losses[-1], steps=len(losses),
                  gate_passed=False, note='Checks differentiability only. Does not open any research phase gate.')


def report(output: Path) -> dict:
    measured = runs()
    lines = ['# LexForm experiment report', '', 'Only completed local runs are included below.', '',
             '| Phase | Run | Status | Micro F1 | Macro F1 |', '|---|---|---|---|---|']
    for item in measured:
        metrics = item.get('metrics', {})
        lines.append(f"| {item['phase']} | {item['id']} | {item['status']} | {metrics.get('micro_f1', '—')} | {metrics.get('macro_f1', '—')} |")
    lines += ['', '## Unfinished phases', '']
    completed = {item['phase'] for item in measured if item['status'] == 'completed'}
    lines += [f'- {phase}: no completed result.' for phase in ('P0', 'P1', 'P2', 'P3', 'P4', 'P5') if phase not in completed]
    lines += ['', '## Interpretation', '',
              'The historical 0.700/0.580 scores are reference targets supplied in the project brief, not newly reproduced results.',
              'Correspondence labels are a mapping-based proxy. Anchored and unanchored transfer must be reported separately.',
              'The handwritten baseline covers only 29 annotated IPC sections and is not directly comparable with all-100 scores.',
              'The depth-two form is an approximation; exceptions, cross-references and nested conditions need legal review.',
              'Cached decomposition token F1 measures lexical coverage, not legal correctness.',
              'Training on court facts does not establish performance on spoken complaints or Tamil–English inputs.',
              'Element-aware prediction and text-described unseen labels are prior art. Novelty is a hypothesis, not established by this implementation.', '',
              '## Related work to verify before publication', '',
              'EDA-NN (ICTAI 2019); CECP (IJCAI 2022); CEEN (2021); Do Charge Prediction Models Learn Legal Theory? (EMNLP 2022);',
              'Santosh, Ichim & Grabmair (EACL 2023); LeSICiN (AAAI 2022); NyayGraph (NLLP 2025).',
              'These references were supplied by the build brief; this generated report is not a completed literature review.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return record(output, 'P6', 'completed', report_file='report.md', result_count=len(measured),
                  all_experiments_completed=all(phase in completed for phase in ('P0', 'P1', 'P2', 'P3', 'P4', 'P5')))


def main() -> None:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    parser = argparse.ArgumentParser(description='Local, gated LexForm research experiments')
    parser.add_argument('command', choices=['setup', 'features', 'bundle', 'audit', 'baseline', 'diagnose', 'decompose', 'train', 'ablate', 'calibrate', 'transfer', 'report', 'smoke'])
    parser.add_argument('--data-dir', type=Path, default=data_dir())
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--gguf', type=Path)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--assets', choices=['all', 'dataset', 'encoder', 'decomposition'], default='all')
    parser.add_argument('--dataset-source', choices=['zenodo', 'sigr-supplement'], default='zenodo')
    args = parser.parse_args()
    if args.checkpoint and not args.config and (args.checkpoint / 'checkpoint.json').is_file():
        from .config import Config
        cfg = Config(**json.loads((args.checkpoint / 'checkpoint.json').read_text())['config'])
    else:
        cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    output = run_path(args.run_id)
    phases = {'baseline': 'P0', 'diagnose': 'P1', 'decompose': 'P2', 'train': 'P3', 'ablate': 'P3-suite',
              'calibrate': 'P4', 'transfer': 'P5', 'report': 'P6', 'smoke': 'smoke', 'audit': 'audit', 'setup': 'setup', 'features': 'features', 'bundle': 'bundle'}
    phase = phases[args.command]
    record(output, phase, 'running', config=cfg.to_dict())
    try:
        if args.command == 'setup':
            from .setup import import_supplement, prepare
            if args.dataset_source == 'sigr-supplement':
                if args.assets != 'dataset':
                    raise ValueError('Use --assets dataset to import the supplemental corpus explicitly')
                preparation = {'assets': import_supplement(args.data_dir), 'errors': {}}
            else:
                preparation = prepare(args.data_dir, args.assets)
            result = record(output, 'setup', 'completed' if not preparation['errors'] else 'failed', **preparation)
            if preparation['errors']:
                print(json.dumps(result, indent=2))
                raise SystemExit(2)
        elif args.command == 'audit':
            result = record(output, 'audit', 'completed', audit=audit(args.data_dir), benchmark=False)
        elif args.command == 'smoke':
            result = smoke(output)
        elif args.command == 'features':
            from .experiments import prepare_features
            result = prepare_features(args.data_dir, cfg, output)
        elif args.command == 'bundle':
            from .bundle import bundle
            if not args.checkpoint:
                raise ValueError('--checkpoint is required')
            result = record(output, 'bundle', 'completed', bundle=bundle(args.checkpoint))
        elif args.command == 'report':
            result = report(output)
        else:
            from .experiments import baseline, calibration, diagnosis, train, transfer
            if args.command == 'baseline':
                result = baseline(args.data_dir, cfg, output)
            else:
                p0 = require_baseline(cfg.protocol) if cfg.protocol == 'canonical' else None
                if cfg.protocol == 'canonical' and args.command in {'decompose', 'train', 'ablate'} and not any(item['phase'] == 'P1' and item['status'] == 'completed'
                        and item.get('dataset_sha256') == p0['manifest']['sha256'] for item in runs()):
                    raise ValueError('P1 is incomplete: execute the encoder diagnosis before decomposition or LexForm training.')
                if args.command == 'diagnose':
                    result = diagnosis(args.data_dir, cfg, output)
                elif args.command == 'decompose':
                    if not args.gguf:
                        raise ValueError('--gguf must point to an existing local quantized model')
                    from .decomposition import decompose
                    result = record(output, 'P2', 'completed', config=cfg.to_dict(), **decompose(args.data_dir, cfg, args.gguf, args.gpu_layers))
                elif args.command == 'train':
                    result = train(args.data_dir, cfg, output)
                elif args.command == 'ablate':
                    from dataclasses import replace
                    experiments = []
                    for variant in ('flat', 'conjunction', 'and_or', 'per_section', 'handwritten'):
                        experiment = train(args.data_dir, replace(cfg, variant=variant), run_path(f'{output.name}-{variant}'))
                        experiments.append({'id': experiment['id'], 'variant': variant,
                                            'metrics': experiment['metrics'], 'label_scope': experiment['label_scope']})
                    result = record(output, 'P3-suite', 'completed', config=cfg.to_dict(), experiments=experiments)
                elif args.command in {'calibrate', 'transfer'}:
                    if not args.checkpoint or not args.checkpoint.is_dir():
                        raise ValueError('--checkpoint must point to an existing completed training run')
                    result = calibration(args.checkpoint, output, cfg) if args.command == 'calibrate' else transfer(args.data_dir, args.checkpoint, output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        status = 'blocked' if isinstance(exc, (FileNotFoundError, ValueError, ImportError)) else 'failed'
        result = record(output, phase, status, config=cfg.to_dict(), error=f'{type(exc).__name__}: {exc}', gate_passed=False)
        print(json.dumps(result, indent=2), flush=True)
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()
