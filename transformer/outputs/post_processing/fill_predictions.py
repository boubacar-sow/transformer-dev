from collections import defaultdict
from functools import lru_cache

from transformer.utils.times import gm_time_to_local_date


@lru_cache(maxsize=2)
def get_daily_right_prs(clean_sillons_getter, day, add_obs_keys=False, filter_only_obs=False):
    """
    Get the complete list or prs that should be predicted on a day
    Also format this list
    """
    clean_sillons = clean_sillons_getter(day)
    res = {}
    keys = {"obs_type", "train_id", "train_num", "pr_cich", "pr_id", "uid"}
    if add_obs_keys:
        keys |= {"delay", "obs_sec", "theo_sec"}
    for train_id, sillon in clean_sillons.items():
        sill = []
        for obs in sillon:
            if filter_only_obs and ("obs_sec" not in obs):
                continue
            sill.append({k: v for k, v in obs.items() if k in keys})
        if len(sill) > 0:
            res[train_id] = sill
    return res


def get_right_prs(
    clean_sillons_getter, timestamp, add_obs_keys=False, filter_only_obs=False, prev_margin=6 * 60, foll_margin=12 * 60
):
    """
    Get the complete list or prs that should be predicted at 'timestamp'
    """
    day = gm_time_to_local_date(timestamp)[:10]
    daily_right_prs = get_daily_right_prs(
        clean_sillons_getter, day, add_obs_keys=add_obs_keys, filter_only_obs=filter_only_obs
    )
    res = {}
    for train_id, sillon in daily_right_prs.items():
        if sillon[0]["theo_sec"] - prev_margin * 60 > timestamp:  # optimization
            continue
        if sillon[-1]["theo_sec"] + foll_margin * 60 < timestamp:  # optimization
            continue
        if ("obs_sec" in sillon[-1]) and (sillon[-1]["obs_sec"] <= timestamp):
            continue
        res[train_id] = [obs.copy() for obs in sillon]
        # le = len(sillon)
        # for i in range(le - 1, -1, -1):
        #     if ("obs_sec" in sillon[i]) and (sillon[i]["obs_sec"] <= timestamp):
        #         i += 1
        #         break
        # if i != le:
        #     res[train_id] = [obs.copy() for obs in sillon[i:]]
    return (timestamp, res)


def set_prediction_on_right_prs(right_prs, predictions):
    """
    must be fmt precise
    After this functions, whatever is in predictions, the predictions returned has exactly one element
        per pr to predict
    Also add truth
    """
    # dict uid to prediction
    preds = {}
    for train_id, pred in predictions[1].items():
        if train_id in right_prs[1]:
            for p in pred:
                d = {k: v for k, v in p.items() if k.startswith("prediction_")}
                preds[p["uid"]] = d
                preds[p["uid"].rsplit("_", 1)[0]] = d
    for train_id, sillon in right_prs[1].items():
        for event in sillon:
            if event["uid"] in preds:
                event.update(preds[event["uid"]])
            elif event["uid"].rsplit("_", 1)[0] in preds:
                event.update(preds[event["uid"].rsplit("_", 1)[0]])
    return right_prs


class FillPredictionsMissingPrs:
    """
    When a prediction algorithm does not predicts some prs, or does not predict every minute,
    this class fill in the blanks with translation
    Also removes predictions on prs that were not hit by the circulation
    """

    def __init__(self, predictions_keys, default_prediction_key="prediction_rttransl"):
        self.predictions_keys = predictions_keys
        self.default_prediction_key = default_prediction_key
        self.last_predictions = defaultdict(dict)

    def __call__(self, predictions):
        current_predictions = defaultdict(dict)
        for train_id, preds in predictions[1].items():
            le = len(preds)
            for max_ipred in range(le - 1, -1, -1):
                if ("obs_sec" in preds[max_ipred]) and (preds[max_ipred]["obs_sec"] <= predictions[0]):
                    max_ipred += 1
                    break
            # First, we fill missing prs by only using the same prediction name,
            # or the last prediction with the same name
            # Missing prs are filled with translation
            for pkey in self.predictions_keys:
                has_one_pred = any(pkey in pred for pred in preds)
                has_one_old_pred = (not has_one_pred) and (
                    any(pred["uid"] in self.last_predictions[pkey] for pred in preds)
                    or any(pred["uid"].rsplit("_", 1)[0] in self.last_predictions[pkey] for pred in preds)
                )
                last_pred = 0
                for ipred, pred in enumerate(preds):
                    if pkey in pred:  # If we find the prediction, we use it
                        last_pred = pred[pkey]
                    elif has_one_pred:  # If a pred. is made, but not on this pr, we fill with transl.
                        if ipred >= max_ipred:
                            pred[pkey] = last_pred
                    elif has_one_old_pred:  # If an old prediction is found, we use it
                        if pred["uid"] in self.last_predictions[pkey]:
                            if ipred >= max_ipred:
                                pred[pkey] = self.last_predictions[pkey][pred["uid"]]
                            last_pred = self.last_predictions[pkey][pred["uid"]]
                        elif pred["uid"].rsplit("_", 1)[0] in self.last_predictions[pkey]:
                            if ipred >= max_ipred:
                                pred[pkey] = self.last_predictions[pkey][pred["uid"].rsplit("_", 1)[0]]
                            last_pred = self.last_predictions[pkey][pred["uid"].rsplit("_", 1)[0]]
                        else:  # and fill its missing prs with translation
                            if ipred >= max_ipred:
                                pred[pkey] = last_pred
                    if pkey in pred:
                        current_predictions[pkey][pred["uid"]] = pred[pkey]
                        current_predictions[pkey][pred["uid"].rsplit("_", 1)[0]] = pred[pkey]
            # Then we fill with 'default_prediction_key' and 0 if default_prediction_key is not found
            for pkey in self.predictions_keys:
                last_pred = 0
                for ipred, pred in enumerate(preds):
                    if pkey in pred:
                        break
                    elif self.default_prediction_key in pred:
                        if ipred >= max_ipred:
                            pred[pkey] = pred[self.default_prediction_key]
                        last_pred = pred[self.default_prediction_key]
                    else:
                        if ipred >= max_ipred:
                            pred[pkey] = last_pred
        self.last_predictions = current_predictions
        return predictions
