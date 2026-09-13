"""PF 1.7 supported index filtering: explicit encoder/decoder validity."""
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    from temporal_protocol import TARGET_OBSERVED, validate_observation_indicator
else:
    from .temporal_protocol import TARGET_OBSERVED, validate_observation_indicator


def validate_dense_frame(frame):
    validate_observation_indicator(frame)
    for _, group in frame.groupby("subject_id", observed=True):
        if "source_split" in group and group.source_split.nunique() != 1:
            raise ValueError("Dataset must contain one split per patient")
        ts = pd.to_datetime(group.timestamp)
        steps = group.time_idx.to_numpy()
        if len(steps) > 1 and (not (np.diff(steps) == 1).all() or
                              not (ts.diff().iloc[1:] == pd.Timedelta("5min")).all()):
            raise ValueError("Dataset requires complete five-minute timeline; no synthetic timesteps")


def window_validity(frame, decoded):
    """Count target occurrences (overlapping windows counted separately)."""
    validate_dense_frame(frame)
    valid_encoder = np.zeros(len(decoded), dtype=bool)
    valid_decoder = np.zeros(len(decoded), dtype=bool)
    missing_targets = np.zeros(len(decoded), dtype=np.int64)
    sizes = (decoded.time_idx_last - decoded.time_idx_first_prediction + 1).to_numpy()
    for subject, windows in decoded.groupby("subject_id", observed=True):
        group = frame[frame.subject_id.astype(str) == str(subject)]
        if group.empty:
            raise ValueError("Unknown patient in dataset index")
        offset = int(group.time_idx.iloc[0])
        start = windows.time_idx_first.to_numpy(dtype=int) - offset
        decoder = windows.time_idx_first_prediction.to_numpy(dtype=int) - offset
        end = windows.time_idx_last.to_numpy(dtype=int) - offset + 1
        if (start < 0).any() or (end > len(group)).any() or (decoder <= start).any():
            raise ValueError("Window crosses supplied patient/split boundary")
        enc_missing = np.r_[0, np.cumsum(~np.isfinite(group.glucose_mg_dl.to_numpy()))]
        dec_missing = np.r_[0, np.cumsum(~group[TARGET_OBSERVED].to_numpy(dtype=bool))]
        positions = decoded.index.get_indexer(windows.index)
        valid_encoder[positions] = (enc_missing[decoder] - enc_missing[start]) == 0
        missing_targets[positions] = dec_missing[end] - dec_missing[decoder]
        valid_decoder[positions] = missing_targets[positions] == 0
    keep = valid_encoder & valid_decoder
    stats = {
        "candidate_windows": int(len(decoded)),
        "retained_windows": int(keep.sum()),
        "excluded_windows": int((~keep).sum()),
        "excluded_window_fraction": float((~keep).mean()),
        "windows_invalid_encoder": int((~valid_encoder).sum()),
        "windows_invalid_decoder": int((~valid_decoder).sum()),
        "candidate_target_occurrences": int(sizes.sum()),
        "nonobserved_target_occurrences": int(missing_targets.sum()),
        "nonobserved_target_fraction": float(missing_targets.sum() / sizes.sum()),
        "retained_target_occurrences": int(sizes[keep].sum()),
        "excluded_target_occurrences": int(sizes[~keep].sum()),
        "zero_weighted_target_occurrences": 0,
    }
    return keep, stats


def filter_observed_windows(dataset, frame):
    keep, stats = window_validity(frame, dataset.decoded_index)
    dataset.filter(lambda _: keep, copy=False)
    dataset.stage_a_window_stats = stats
    return dataset


def assert_observed_evaluation(dataset, frame):
    """Fail before prediction if an unfiltered/legacy dataset is supplied."""
    keep, _ = window_validity(frame, dataset.decoded_index)
    if not keep.all():
        raise ValueError("Evaluation contains missing encoder or unobserved decoder targets")
