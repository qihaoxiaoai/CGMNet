# cgmnet/training/evaluator.py
"""
A comprehensive evaluator for computing various performance metrics for both
classification and regression tasks.
"""
import numpy as np
import torch
from typing import Dict, List, Callable
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, r2_score, f1_score

def rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def f1(y_true, y_pred):
    """Calculates F1 score after thresholding predictions at 0.5."""
    # This assumes binary classification with logits. Sigmoid would map 0 to 0.5.
    y_pred_class = (y_pred > 0).astype(int)
    return f1_score(y_true, y_pred_class)

METRIC_FUNC_MAP: Dict[str, Callable] = {
    'auc': roc_auc_score,
    'ap': average_precision_score,
    'f1': f1,
    'rmse': rmse,
    'mae': mean_absolute_error,
    'r2': r2_score,
}

class Evaluator:
    """Computes a list of scores for a given set of predictions and labels."""
    def __init__(self, metrics: List[str], n_tasks: int):
        self.metrics = metrics
        self.n_tasks = n_tasks
        
        for metric in self.metrics:
            if metric not in METRIC_FUNC_MAP:
                raise ValueError(f"Metric '{metric}' not supported. Available: {list(METRIC_FUNC_MAP.keys())}")

    def _parse_input(self, y_true, y_pred):
        """Converts tensors to numpy arrays and validates shapes."""
        if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
        return y_true, y_pred

    def eval(self, y_true, y_pred) -> Dict[str, float]:
        """
        Calculates all requested metrics and returns them in a dictionary.
        """
        y_true, y_pred = self._parse_input(y_true, y_pred)
        
        results = {}
        for metric in self.metrics:
            metric_func = METRIC_FUNC_MAP[metric]
            task_scores = []
            for i in range(self.n_tasks):
                task_true = y_true[:, i]
                task_pred = y_pred[:, i]
                
                # Mask out NaN values for the current task
                is_labeled = ~np.isnan(task_true)
                if not np.any(is_labeled):
                    continue

                true_masked, pred_masked = task_true[is_labeled], task_pred[is_labeled]
                
                # For AUC, ensure both classes are present
                if metric == 'auc' and len(np.unique(true_masked)) < 2:
                    continue
                    
                task_scores.append(metric_func(true_masked, pred_masked))
            
            # Average the scores across all valid tasks
            results[metric] = np.nanmean(task_scores) if task_scores else 0.0
            
        return results
