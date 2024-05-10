def round_prediction_fmt_precise(predictions, step=1, shift=0.0):
    """
    x --> int((x + shift)//step) * step
    """
    for preds in predictions.values():
        for pred in preds:
            for k in pred.keys():
                if k.startswith("prediction_"):
                    pred[k] = int((pred[k] + shift) // step) * step
    return predictions


def round_prediction_fmt_approx(predictions, step=1, shift=0.0):
    """
    x --> int((x + shift)//step) * step
    """
    for preds in predictions.values():
        for k in preds.keys():
            if k.startswith("prediction_"):
                preds[k] = [int((pred + shift) // step) * step for pred in preds[k]]
    return predictions
