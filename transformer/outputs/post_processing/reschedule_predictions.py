from transformer.utils.times import (gm_time_to_local_date,
                                     local_date_to_gm_time)


def reschedule_predictions_every(
    pred_generator,
    start_margin=3 * 60,
    end_margin=3 * 60,
    start_time=None,
    end_time=None,
    every=60,
    modulo=0,
    verbose=True,
):
    """
    Reschedule predictions so that there are one prediction every 'every' seconds
    """
    last_time_pred = None
    # time of first prediction
    if start_time is not None:
        if isinstance(start_time, str):
            start_time = local_date_to_gm_time(start_time)
        last_time_pred = start_time
    # time of last prediction (excluded)
    if end_time is not None:
        if isinstance(end_time, str):
            end_time = local_date_to_gm_time(end_time)
    # For every prediction received
    is_before_start = True
    if verbose:
        print(
            "start_time",
            (start_time if (start_time is None) else gm_time_to_local_date(start_time)),
            "end_time",
            (end_time if (end_time is None) else gm_time_to_local_date(end_time)),
        )
    current_prediction = {}
    for time_pred, pred in pred_generator:
        if verbose:
            print(
                "Received prediction at: ",
                gm_time_to_local_date(time_pred),
                "(len_{})".format(len(pred)),
                is_before_start,
            )
        if is_before_start and (start_time is not None) and (time_pred < start_time - 12 * 60 * 60):
            if verbose:
                print("Computing skipped: more than 12 hours before start time")
            continue
        if last_time_pred is None:
            last_time_pred = time_pred - start_margin * every
        # We add metrics only if a prediction of the next minute came from pred_generator
        if int((time_pred - modulo) // every) > int((last_time_pred - modulo) // every):
            # add one or several layer of predictions
            last_tp = int((last_time_pred - modulo) // every + 1) * every + modulo
            tp = int((time_pred - modulo) // every + 1) * every + modulo
            for timestamp in range(last_tp, tp, every):
                if (start_time is not None) and (timestamp >= start_time) and is_before_start:
                    is_before_start = False
                if (end_time is None) or (timestamp < end_time):
                    yield [timestamp, current_prediction, is_before_start]
                    current_prediction = {}
                else:
                    break
            if (end_time is not None) and (timestamp >= end_time):
                break
        last_time_pred = time_pred
        current_prediction = pred
    time_pred = time_pred + (end_margin + 1) * every
    last_tp = int((last_time_pred - modulo) // every + 1) * every + modulo
    tp = int((time_pred - modulo) // every + 1) * every + modulo
    for timestamp in range(last_tp, tp, every):
        if (start_time is not None) and (timestamp >= start_time) and is_before_start:
            is_before_start = False
        if (end_time is None) or (timestamp < end_time):
            yield [timestamp, current_prediction, is_before_start]
            current_prediction = {}
        else:
            break
