from transformer.outputs.post_processing.fill_predictions import (
    FillPredictionsMissingPrs, get_right_prs, set_prediction_on_right_prs)
from transformer.outputs.post_processing.round_predictions import (
    round_prediction_fmt_approx, round_prediction_fmt_precise)


def post_processing_fmt_precise(predictions):
    timestamp, preds = predictions
    preds = round_prediction_fmt_precise(preds, step=1, shift=0.0)
    return [timestamp, preds]


def post_processing_fmt_approx(predictions):
    timestamp, preds = predictions
    preds = round_prediction_fmt_approx(preds, step=1, shift=0.0)
    return [timestamp, preds]


class PostProcessingFillMissingPrs:
    """
    Only fmt precise
    """

    def __init__(
        self,
        clean_sillons_getter,
        predictions_keys,
        default_prediction_key="prediction_rttransl",
        add_obs_keys=False,
        filter_only_obs=False,
        prev_margin=6 * 60,
        foll_margin=12 * 60,
    ):
        # self.predictions_keys = predictions_keys
        # self.default_prediction_key = default_prediction_key
        self.clean_sillons_getter = clean_sillons_getter
        self.add_obs_keys = add_obs_keys
        self.filter_only_obs = filter_only_obs
        self.prev_margin = prev_margin
        self.foll_margin = foll_margin
        self.fill_predictions = FillPredictionsMissingPrs(
            predictions_keys, default_prediction_key=default_prediction_key
        )

    def __call__(self, predictions, timestamp):
        right_prs = get_right_prs(
            self.clean_sillons_getter,
            timestamp,
            add_obs_keys=self.add_obs_keys,
            filter_only_obs=self.filter_only_obs,
            prev_margin=self.prev_margin,
            foll_margin=self.foll_margin,
        )
        predictions = set_prediction_on_right_prs(right_prs, predictions)
        predictions = self.fill_predictions(predictions)
        return predictions
