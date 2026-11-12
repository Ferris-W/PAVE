"""ST-FiT metrics on [sample, horizon, node], in original flow units."""
import numpy as np


def RMSE_MAE_MAPE(y_true, y_pred):
    y, pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if y.shape != pred.shape or y.size == 0:
        raise ValueError("Metric arrays must be nonempty with identical shapes")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Nonfinite prediction/target: refusing to hide NaNs in metrics")
    error = pred - y
    mask = y >= 1
    mape = float(np.abs(error[mask] / y[mask]).mean() * 100) if mask.any() else None
    return float(np.sqrt(np.mean(error ** 2))), float(np.mean(np.abs(error))), mape


def metric_dict(y, pred):
    rmse, mae, mape = RMSE_MAE_MAPE(y, pred)
    return {"mae": mae, "rmse": rmse, "mape_percent": mape, "mape_valid_count": int(np.sum(y >= 1))}


def evaluate_groups(y, pred, seen, unseen):
    if y.ndim != 3 or y.shape[1] != 12 or y.shape != pred.shape:
        raise ValueError("Expected matching [sample, 12, node] arrays")
    result = {}
    for name, ids in [("all", np.arange(y.shape[2])), ("seen", seen), ("unseen", unseen)]:
        if len(ids) == 0:
            result[name] = {"prefix": {"12": dict(mae=None, rmse=None,
                mape_percent=None, mape_valid_count=0)}}
            continue
        actual, predicted = y[:, :, ids], pred[:, :, ids]
        result[name] = {
            "prefix": {"12": metric_dict(actual, predicted)},
        }
    return result
