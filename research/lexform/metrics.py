from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def multilabel(y: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        'micro_f1': float(f1_score(y, prediction, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y, prediction, average='macro', zero_division=0)),
        'precision': float(precision_score(y, prediction, average='micro', zero_division=0)),
        'recall': float(recall_score(y, prediction, average='micro', zero_division=0)),
        'labels_used': int(prediction.any(axis=0).sum()),
        'labels_total': int(y.shape[1]),
        'documents': int(len(y)),
    }


def tune_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 37)
    return float(max(candidates, key=lambda threshold: multilabel(y, scores >= threshold)['micro_f1']))


def select(scores: np.ndarray, weakest: np.ndarray, threshold: float, floor: float) -> np.ndarray:
    return (scores >= threshold) & (weakest >= floor)


def selective_point(y: np.ndarray, prediction: np.ndarray) -> dict:
    accepted = prediction.any(axis=1)
    count = int(accepted.sum())
    if count:
        true = y[accepted]
        pred = prediction[accepted]
        intersection = (true & pred).sum(axis=1)
        denominator = true.sum(axis=1) + pred.sum(axis=1)
        risk = float(np.mean(1 - 2 * intersection / np.maximum(denominator, 1)))
    else:
        risk = None
    return {'coverage': float(accepted.mean()), 'risk': risk, 'accepted_documents': count}


def calibrate(y: np.ndarray, scores: np.ndarray, weakest: np.ndarray, target_risk: float, min_accepted: int = 30) -> dict:
    points = []
    for threshold in np.linspace(0.05, 0.95, 19):
        for floor in np.linspace(0, 0.9, 10):
            point = selective_point(y, select(scores, weakest, threshold, floor))
            points.append({**point, 'threshold': float(threshold), 'floor': float(floor)})
    feasible = [point for point in points if point['accepted_documents'] >= min_accepted
                and point['risk'] is not None and point['risk'] <= target_risk]
    chosen = max(feasible, key=lambda point: (point['coverage'], -point['risk'])) if feasible else {
        'threshold': 1.000001, 'floor': 1.0, 'coverage': 0.0, 'risk': None, 'accepted_documents': 0}
    return {**chosen, 'target_risk': target_risk, 'calibration_split': 'dev',
            'risk_definition': 'mean(1 - document F1) among documents with at least one emitted section',
            'feasible': bool(feasible), 'minimum_accepted': min_accepted}


def risk_curve(y: np.ndarray, scores: np.ndarray, weakest: np.ndarray, floor: float) -> list[dict]:
    return [{**selective_point(y, select(scores, weakest, threshold, floor)), 'threshold': float(threshold)}
            for threshold in np.linspace(1, 0, 51)]
