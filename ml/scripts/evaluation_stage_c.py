"""Versioned Stage C evaluation, separate from training/early-stopping indices.

Raw mg/dL, unweighted occurrence metrics. No fitting, calibration or clinical
thresholds. Every published comparison covers the complete pre-prediction W.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib
import json
import logging

import numpy as np
import pandas as pd
import torch

try:
    from .checkpoint_registry import atomic_json, digest, semantic, sha256
    from .numerical_profile import verify as verify_numerics
except ImportError:
    from checkpoint_registry import atomic_json, digest, semantic, sha256
    from numerical_profile import verify as verify_numerics

REVISION = "nmd-evaluation-stage-c-1"
NORMALIZER = "observed-train-encoded-subject-v2"
Q = (.02, .10, .25, .50, .75, .90, .98)
HORIZONS = tuple(range(5, 61, 5))
KEYS = ("subject", "window_id", "origin_timestamp", "target_timestamp", "horizon_minutes")
INTERPRETATION = {
    "scope": "within-subject temporal; conditional on eligibility",
    "calibration": {"status": "raw/not_fitted", "fitted_source": None, "fitted_role": None},
    "C13-F01": "ACCEPTED LIMITATION — WEIGHTING UNCHANGED",
    "weighted_estimand": "Fw(a|x)=E[w(Y)1{Y<=a}|x]/E[w(Y)|x]; raw q.5 need not be the unweighted median",
    "C14-F01": "NOT IMPLEMENTED/DEFERRED",
    "clinical_reliability": "NOT EVALUATED",
    "Clarke_full_geometry": "NOT EVALUATED; legacy interior examples only",
    "dependence": "overlapping forecast occurrences are not IID",
}
log = logging.getLogger(__name__)


class EvaluationError(ValueError):
    def __init__(self, code, stage, message):
        self.code, self.stage = code, stage
        super().__init__(f"{code} at {stage}: {message}")


def require(condition, code, stage, message):
    if not condition:
        raise EvaluationError(code, stage, message)


def finite(value, stage):
    require(np.isfinite(value).all(), "finite", stage, "nonfinite required value or reduction")


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def quantiles(values, width=None):
    require(tuple(values) == Q, "quantiles", "schema", "ordered native Q required")
    if width is not None:
        require(width == len(Q), "quantiles", "output", "output axis disagrees with Q")
    return list(values)


def interval_pair(level):
    require(level in (50, 80, 96), "quantiles", "interval", "unavailable interval endpoints")
    return {50: (2, 4), 80: (1, 5), 96: (0, 6)}[level]


def state_hash(model):
    return state_mapping_hash(model.state_dict())


def state_mapping_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        v = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def source_hash(frame):
    # Source is supplied explicitly; never open a dataset in this module.
    return hashlib.sha256(frame.to_json(orient="table", date_format="iso", double_precision=15).encode()).hexdigest()


def schema(dataset):
    p = dataset.get_parameters()
    names = ("time_idx", "target", "group_ids", "weight", "static_categoricals", "static_reals",
             "time_varying_known_categoricals", "time_varying_known_reals", "time_varying_unknown_categoricals",
             "time_varying_unknown_reals", "variable_groups", "categorical_encoders", "scalers", "lags",
             "add_relative_time_idx", "add_target_scales", "add_encoder_length", "allow_missing_timesteps")
    return semantic({name: p[name] for name in names})


def normalization(dataset):
    norm = dataset.target_normalizer
    require(getattr(norm, "nmd_revision", None) == NORMALIZER, "compatibility", "normalizer", "historical normalizer rejected")
    return semantic(norm)


def code_hashes():
    return {name: sha256(Path(__file__).with_name(name)) for name in (
        "evaluation_stage_c.py", "train_tft_population_v2.py", "checkpoint_registry.py",
        "observed_windows.py", "finite_training.py", "numerical_profile.py")}


@dataclass(frozen=True)
class OwnedSyntheticState:
    """Receipt issued only when this process creates its own untrained state.

    This is local ownership/provenance, not authentication of external files.
    It is deliberately separate from the production BEST registry.
    """
    path: Path
    metadata: dict
    file_sha256: str


def own_synthetic_state(model, dataset, path, *, config, code):
    require(code.get("head") and isinstance(code.get("dirty"), bool), "context", "ownership", "code identity required")
    quantiles(model.loss.quantiles)
    path = Path(path)
    require(not path.exists(), "artifact", "ownership", "immutable destination exists")
    metadata = {"revision": REVISION, "role": "SYNTHETIC_UNTRAINED", "units": "mg/dL",
                "quantiles": list(Q), "schema": schema(dataset), "normalizer": normalization(dataset),
                "state_sha256": state_hash(model), "config": copy.deepcopy(config),
                "config_sha256": digest(config), "code": copy.deepcopy(code), "code_sha256": code_hashes()}
    with path.open("xb") as stream:
        torch.save({"metadata": metadata, "state_dict": model.state_dict(),
                    "dataset_parameters": dataset.get_parameters()}, stream)
    atomic_json(path.with_suffix(".json"), metadata)
    return OwnedSyntheticState(path.resolve(), metadata, sha256(path))


def load_owned_state(receipt, model, dataset, *, expected):
    """Validate readable metadata and bytes BEFORE deserializing owned pickle."""
    require(isinstance(receipt, OwnedSyntheticState), "context", "preload", "owned receipt required")
    sidecar = json.loads(receipt.path.with_suffix(".json").read_text())
    for field in ("quantiles",):
        quantiles(sidecar.get(field, []))
    require(sidecar == receipt.metadata == expected, "compatibility", "preload", "metadata mismatch")
    require(sidecar["revision"] == REVISION and sidecar["role"] == "SYNTHETIC_UNTRAINED", "role", "preload", "wrong state role")
    require(sidecar["schema"] == schema(dataset) and sidecar["normalizer"] == normalization(dataset),
            "compatibility", "preload", "dataset semantics mismatch")
    require(sha256(receipt.path) == receipt.file_sha256, "artifact", "preload", "state bytes mismatch")
    # Only bytes created by own_synthetic_state and validated above are accepted.
    owned = torch.load(receipt.path, map_location="cpu", weights_only=False)
    require(owned["metadata"] == sidecar, "artifact", "load", "embedded metadata mismatch")
    model.load_state_dict(owned["state_dict"], strict=True)
    require(state_hash(model) == sidecar["state_sha256"], "artifact", "load", "loaded state mismatch")
    return owned["dataset_parameters"]


@dataclass(frozen=True)
class RegisteredBestState:
    """Receipt from the existing production ownership/compatibility registry."""
    path: Path
    metadata: dict
    file_sha256: str
    registry: object
    run_id: str
    expected: dict


def registered_best_context(registry, run_id, expected, model, dataset, frame, *, split_role, config, code):
    """Explicit BEST linkage for a separately authorized real evaluation.

    No discovery, loading of data, model fit, or registry bypass. Not exercised
    with trained artifacts by the synthetic-only remediation campaign.
    """
    require(split_role in ("validation", "test"), "role", "preload", "explicit evaluation role required")
    require(expected.get("normalizer_revision") == NORMALIZER, "compatibility", "preload", "historical weights rejected")
    quantiles(expected.get("quantiles", []))
    require(expected.get("normalizer") == normalization(dataset) and expected.get("schema") == schema(dataset),
            "compatibility", "preload", "dataset semantics mismatch")
    path, payload, record = registry.verified(run_id, "best", expected)
    require(state_mapping_hash(payload["state_dict"]) == state_hash(model), "context", "load", "in-memory state is not verified BEST")
    metadata = {"revision": REVISION, "role": "BEST", "units": "mg/dL", "quantiles": list(Q),
                "schema": schema(dataset), "normalizer": normalization(dataset), "state_sha256": state_hash(model),
                "config": copy.deepcopy(config), "config_sha256": digest(config), "code": copy.deepcopy(code),
                "code_sha256": code_hashes(), "training_contract": expected, "registry_run_id": run_id}
    receipt = RegisteredBestState(path, metadata, record["sha256"], registry, run_id, copy.deepcopy(expected))
    context = EvaluationContext(receipt, source_hash(frame), split_role, schema(dataset), normalization(dataset), digest(config))
    validate_context(context, model, dataset, frame)
    return context


@dataclass(frozen=True)
class EvaluationContext:
    receipt: OwnedSyntheticState | RegisteredBestState
    source_sha256: str
    split_role: str
    schema: dict
    normalizer: dict
    config_sha256: str


def synthetic_context(receipt, model, dataset, frame, *, split_role, config):
    require(split_role.startswith("synthetic_"), "role", "context", "synthetic role required")
    context = EvaluationContext(receipt, source_hash(frame), split_role, schema(dataset),
                                normalization(dataset), digest(config))
    validate_context(context, model, dataset, frame)
    return context


def validate_context(context, model, dataset, frame):
    require(isinstance(context, EvaluationContext), "context", "input", "verified evaluation context required")
    receipt = context.receipt
    require(isinstance(receipt, (OwnedSyntheticState, RegisteredBestState)), "context", "input", "owned model required")
    m = receipt.metadata
    quantiles(m.get("quantiles", [])); quantiles(model.loss.quantiles)
    require(model.hparams.output_size == len(Q), "quantiles", "input", "model output width mismatch")
    require(m.get("revision") == REVISION, "compatibility", "input", "historical revision")
    synthetic = isinstance(receipt, OwnedSyntheticState)
    require((synthetic and m.get("role") == "SYNTHETIC_UNTRAINED" and context.split_role.startswith("synthetic_")) or
            (not synthetic and m.get("role") == "BEST" and context.split_role in ("validation", "test")), "role", "input", "wrong role")
    require(m.get("units") == "mg/dL", "schema", "input", "native mg/dL required")
    require(m["code_sha256"] == code_hashes(), "context", "input", "source code changed")
    require(context.config_sha256 == m["config_sha256"] == digest(m["config"]), "context", "input", "config mismatch")
    require(context.schema == m["schema"] == schema(dataset), "schema", "input", "dataset schema mismatch")
    require(context.normalizer == m["normalizer"] == normalization(dataset), "compatibility", "input", "normalizer mismatch")
    require(semantic(model.output_transformer) == context.normalizer, "compatibility", "input", "model inverse differs from dataset")
    require(source_hash(frame) == context.source_sha256, "context", "input", "source changed")
    require(sha256(receipt.path) == receipt.file_sha256, "artifact", "input", "owned artifact changed")
    if synthetic:
        require(json.loads(receipt.path.with_suffix(".json").read_text()) == m, "artifact", "input", "sidecar changed")
    else:
        _, payload, record = receipt.registry.verified(receipt.run_id, "best", receipt.expected)
        require(record["sha256"] == receipt.file_sha256 and state_mapping_hash(payload["state_dict"]) == m["state_sha256"],
                "artifact", "input", "BEST receipt changed")
    require(state_hash(model) == m["state_sha256"], "context", "input", "in-memory model changed")
    require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()),
            "context", "input", "evaluation profile requires CPU FP32")
    verify_numerics("cpu")


def qualify(dataset, frame, role):
    """G1a: immutable canonical W/S; never mutate Stage A training index."""
    required = {"subject_id", "time_idx", "timestamp", "source_split", "target_observed", "glucose_mg_dl"}
    require(required.issubset(frame), "schema", "source", "missing source fields")
    require(frame.target_observed.isin([True, False]).all(), "schema", "source", "invalid observation flag")
    f = frame.copy(); f["timestamp"] = pd.to_datetime(f.timestamp)
    require(not f.duplicated(["subject_id", "time_idx"]).any() and
            not f.duplicated(["subject_id", "timestamp"]).any(), "alignment", "source", "duplicate source keys")
    lookup = {}
    for sid, g in f.groupby("subject_id", sort=True):
        g = g.sort_values("time_idx").reset_index(drop=True)
        require((np.diff(g.time_idx) == 1).all() and
                (g.timestamp.diff().iloc[1:] == pd.Timedelta(minutes=5)).all(),
                "alignment", "source", "irregular actual timeline")
        observed = g.target_observed.to_numpy(bool)
        finite(g.glucose_mg_dl.to_numpy()[observed], "source target")
        require((g.glucose_mg_dl.to_numpy()[observed] > 0).all(), "target_domain", "source", "observed target must be positive")
        lookup[str(sid)] = g.set_index("time_idx", drop=False)
    decoded = dataset.decoded_index
    keep, windows, exclusions = [], [], {"length": 0, "unobserved_decoder": 0, "encoder": 0}
    for row in decoded.itertuples(index=False):
        sid = str(row.subject_id); first, dec, last = map(int, (row.time_idx_first, row.time_idx_first_prediction, row.time_idx_last))
        if dec - first != 48 or last - dec != 11:
            keep.append(False); exclusions["length"] += 1; continue
        require(sid in lookup, "alignment", "windows", "unknown subject")
        g = lookup[sid]
        require(set(range(first, last + 1)).issubset(g.index), "alignment", "windows", "window outside source")
        enc, target = g.loc[first:dec-1], g.loc[dec:last]
        require((target.source_split == role).all(), "role", "windows", "decoder crosses role boundary")
        if not target.target_observed.all():
            keep.append(False); exclusions["unobserved_decoder"] += 1; continue
        valid = np.isfinite(enc.glucose_mg_dl).all()
        for i, e in enc.iterrows():
            if not e.target_observed:
                past = g.loc[:i]; past = past[past.target_observed]
                valid = valid and len(past) > 0
                if len(past):
                    source = past.iloc[-1]
                    valid = valid and 0 < i - int(source.time_idx) <= 6 and e.glucose_mg_dl == source.glucose_mg_dl
        if not valid:
            keep.append(False); exclusions["encoder"] += 1; continue
        origin = enc.timestamp.iloc[-1].isoformat()
        windows.append({"subject": sid, "window_id": sid + ":" + origin, "origin_timestamp": origin,
                        "first": first, "decoder": dec, "last": last,
                        "targets": target.glucose_mg_dl.to_numpy(float).tolist(),
                        "target_timestamps": [t.isoformat() for t in target.timestamp],
                        "encoder_end": float(enc.glucose_mg_dl.iloc[-1])})
        keep.append(True)
    windows.sort(key=lambda w: (w["subject"], w["origin_timestamp"]))
    require(len({w["window_id"] for w in windows}) == len(windows), "alignment", "windows", "duplicate window")
    keys = [{k: w[k] for k in KEYS[:3]} | {"target_timestamp": w["target_timestamps"][i], "horizon_minutes": h}
            for w in windows for i, h in enumerate(HORIZONS)]
    for key in keys:
        require(pd.Timestamp(key["target_timestamp"]) == pd.Timestamp(key["origin_timestamp"]) + pd.Timedelta(minutes=key["horizon_minutes"]),
                "alignment", "windows", "horizon timestamp mismatch")
    # PF filter refuses an empty index: no loader is needed for empty W.
    selected = dataset.filter(lambda _: np.asarray(keep, bool)) if windows else None
    return selected, windows, keys, {"candidate_index_rows": len(decoded), "eligible_windows": len(windows), "excluded": exclusions}


def persistence(windows, frame):
    values, lineage = [], []
    for w in windows:
        origin = pd.Timestamp(w["origin_timestamp"])
        start = origin - pd.Timedelta(minutes=47*5)
        enc = frame[(frame.subject_id.astype(str) == w["subject"]) &
                    pd.to_datetime(frame.timestamp).between(start, origin)].sort_values("timestamp")
        observed = enc[enc.target_observed.astype(bool)]
        require(len(enc) == 48 and not observed.empty, "persistence", "baseline", "missing source in encoder")
        source = observed.iloc[-1]; timestamp = pd.Timestamp(source.timestamp)
        age = (origin - timestamp) / pd.Timedelta(minutes=5)
        value = float(source.glucose_mg_dl)
        require(np.isfinite(value) and value > 0 and age == int(age) and 0 <= age <= 6,
                "persistence", "baseline", "invalid observation age/value")
        require(value == w["encoder_end"] == float(enc.glucose_mg_dl.iloc[-1]),
                "persistence", "baseline", "source and encoder end disagree")
        values.append(value)
        lineage.append({"window_id": w["window_id"], "subject": w["subject"], "source_timestamp": timestamp.isoformat(),
                        "origin_timestamp": w["origin_timestamp"], "age_bins": int(age), "ffill": bool(age), "value_mg_dl": value})
    return np.repeat(np.asarray(values, float), 12), lineage


def align(predictions, dataset, windows):
    x = predictions.x
    n = len(windows)
    require(all(k in x for k in ("decoder_time_idx", "encoder_lengths", "decoder_lengths", "groups", "encoder_target")),
            "alignment", "return", "missing prediction index fields")
    require(array(x["encoder_lengths"]).shape == (n,) and array(x["decoder_lengths"]).shape == (n,), "alignment", "return", "length count mismatch")
    require((array(x["encoder_lengths"]) == 48).all() and (array(x["decoder_lengths"]) == 12).all(), "length", "return", "full 48/12 required")
    y = predictions.y[0] if isinstance(predictions.y, (tuple, list)) else predictions.y
    y = array(y).astype(np.float64)
    z = array(predictions.output).astype(np.float64)
    require(z.ndim == 3, "alignment", "return", "prediction shape")
    quantiles(Q, z.shape[-1])
    require(z.shape == (n, 12, 7) and y.shape == (n, 12), "alignment", "return", "target/output shape mismatch")
    finite(y, "returned target"); finite(z, "returned quantiles")
    require((y > 0).all(), "target_domain", "return", "target must be positive")
    idx = dataset.x_to_index(x)
    times = array(x["decoder_time_idx"])
    require(times.shape == (n, 12), "alignment", "return", "decoder time shape")
    order = {(w["subject"], w["decoder"]): i for i, w in enumerate(windows)}
    seen, reordered_y, reordered_z = set(), np.empty_like(y), np.empty_like(z)
    for j, row in enumerate(idx.itertuples(index=False)):
        key = (str(row.subject_id), int(row.time_idx))
        require(key in order and key not in seen, "alignment", "return", "extra or duplicate key")
        seen.add(key); i = order[key]; w = windows[i]
        require(np.array_equal(times[j], np.arange(w["decoder"], w["last"]+1)), "alignment", "return", "horizon keys mismatch")
        require(np.allclose(y[j].astype(np.float32), np.asarray(w["targets"], np.float32), atol=1e-5, rtol=1e-5),
                "alignment", "return", "returned target disagrees with source")
        require(np.isclose(array(x["encoder_target"])[j, -1], np.float32(w["encoder_end"]), atol=1e-5, rtol=1e-5),
                "persistence", "encoder", "actual TFT encoder end differs from source")
        reordered_y[i], reordered_z[i] = y[j], z[j]
    require(seen == set(order), "alignment", "return", "missing keys")
    return reordered_y.reshape(-1), reordered_z.reshape(-1, 7)


def reduce_metrics(y, p, z=None):
    """FP64 full-precision reporting primitives; no sample selection."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    require(y.ndim == 1 and y.shape == p.shape, "alignment", "metrics", "point shapes")
    finite(y, "metric target"); finite(p, "metric point")
    require((y > 0).all(), "target_domain", "metrics", "target must be positive")
    n = len(y)
    result = {"N": n, "status": "VALID" if n else "EMPTY", "reason": None if n else "no occurrences",
              "point": {k: None for k in ("mae", "rmse", "bias", "mard")},
              "nonpositive_predictions": int((p <= 0).sum()), "probabilistic": {"status": "N/A", "reason": "point-only baseline"}}
    if n:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            e = p-y
            values = [np.abs(e).mean(), np.sqrt(np.square(e).mean()), e.mean(), 100*np.mean(np.abs(e)/y)]
        finite(values, "point reduction")
        result["point"] = dict(zip(result["point"], map(float, values)))
    if z is not None:
        z = np.asarray(z, float)
        require(z.shape == (n, 7), "quantiles", "metrics", "quantile shape")
        finite(z, "metric quantiles")
        prob = {"status": "VALID" if n else "EMPTY", "quantiles": {}, "intervals": {},
                "crossing": {"any_rate": None, "adjacent_rate": None, "max_magnitude": None,
                             "N_any": 0, "N_adjacent": 0, "adjacent_denominator": n*6}}
        for j, q in enumerate(Q):
            hit = float(np.mean(y <= z[:, j])) if n else None
            with np.errstate(over="ignore", invalid="ignore"):
                u = y-z[:, j]
                pin = float(np.mean(u*(q-(u < 0)))) if n else None
            if n: finite(pin, "pinball reduction")
            prob["quantiles"][str(q)] = {"pinball": pin, "hit": hit, "hit_minus_q": hit-q if n else None,
                                        "hit_minus_q_pp": 100*(hit-q) if n else None, "N": n}
        for level in (50, 80, 96):
            a, b = interval_pair(level); lo, hi = z[:, a], z[:, b]
            bad = int((lo > hi).sum())
            width = None
            if n and not bad:
                with np.errstate(over="ignore", invalid="ignore"):
                    width = float(np.mean(hi-lo))
                finite(width, "interval reduction")
            prob["intervals"][str(level)] = {"status": "INVALID" if bad else "VALID" if n else "EMPTY",
                "N": n, "N_invalid": bad, "picp": float(np.mean((lo <= y) & (y <= hi))) if n and not bad else None,
                "width": width, "nominal": level/100}
        if n:
            with np.errstate(over="ignore", invalid="ignore"):
                differences = z[:, :-1]-z[:, 1:]
            finite(differences, "crossing reduction")
            crossed = differences > 0
            prob["crossing"] = {"any_rate": float(crossed.any(1).mean()), "adjacent_rate": float(crossed.mean()),
                "max_magnitude": float(max(0., differences.max())), "N_any": int(crossed.any(1).sum()),
                "N_adjacent": int(crossed.sum()), "adjacent_denominator": n*6}
        result["probabilistic"] = prob
    return result


def denominators(keys):
    return {"occurrences": len(keys), "windows": len({k["window_id"] for k in keys}),
            "unique_subject_target_timestamps": len({(k["subject"], k["target_timestamp"]) for k in keys}),
            "subjects": len({k["subject"] for k in keys})}


def macro(metrics, subjects):
    included = [s for s in subjects if metrics[s]["N"]]
    present = [metrics[s] for s in included]
    # Start with full pooled denominators supplied by caller; metrics below are
    # equal-patient means, not an average of batch means or patient MSEs.
    result = copy.deepcopy(metrics[subjects[0]]) if subjects else reduce_metrics([], [])
    result.pop("denominators", None)
    result.update(N=sum(m["N"] for m in present), status="VALID" if present else "EMPTY",
                  reason=None if present else "no occurrences", included_subjects=included,
                  missing_subjects=[s for s in subjects if s not in included], subjects_total=len(subjects))
    mean = lambda values: float(np.mean(values)) if values else None
    result["point"] = {k: mean([m["point"][k] for m in present]) for k in result["point"]}
    result["nonpositive_predictions"] = sum(m["nonpositive_predictions"] for m in present)
    prob = result["probabilistic"]
    if prob["status"] != "N/A":
        prob["status"] = "VALID" if present else "EMPTY"
        for q, cell in prob["quantiles"].items():
            for key in ("pinball", "hit", "hit_minus_q", "hit_minus_q_pp"):
                cell[key] = mean([m["probabilistic"]["quantiles"][q][key] for m in present])
            cell["N"] = result["N"]
        for level, cell in prob["intervals"].items():
            parts = [m["probabilistic"]["intervals"][level] for m in present]
            bad = sum(p["N_invalid"] for p in parts)
            cell.update(N=result["N"], N_invalid=bad, status="INVALID" if bad else "VALID" if present else "EMPTY")
            for key in ("picp", "width"):
                cell[key] = mean([p[key] for p in parts]) if not bad else None
        cross = prob["crossing"]
        for key in ("any_rate", "adjacent_rate"):
            cross[key] = mean([m["probabilistic"]["crossing"][key] for m in present])
        cross["max_magnitude"] = mean([m["probabilistic"]["crossing"]["max_magnitude"] for m in present])
        for key in ("N_any", "N_adjacent", "adjacent_denominator"):
            cross[key] = sum(m["probabilistic"]["crossing"][key] for m in present)
    return result


def report_metrics(keys, y, z, baseline, subjects):
    y, z, baseline = np.asarray(y, float), np.asarray(z, float), np.asarray(baseline, float)
    require(len(keys) == len(y) == len(baseline) == len(z), "alignment", "report", "key cardinality")
    subjects = sorted(set(subjects)); ids = np.array([k["subject"] for k in keys]); hs = np.array([k["horizon_minutes"] for k in keys])
    strata = {"all": np.ones(len(y), bool), "three:lt70": y < 70, "three:70to180": (y >= 70) & (y <= 180),
              "three:gt180": y > 180, "five:lt54": y < 54, "five:54to70": (y >= 54) & (y < 70),
              "five:70to180": (y >= 70) & (y <= 180), "five:180to250": (y > 180) & (y <= 250), "five:gt250": y > 250}
    panels = {}
    for horizon in ("all", *HORIZONS):
        horizon_mask = np.ones(len(y), bool) if horizon == "all" else hs == horizon
        ranges = {}
        for name, range_mask in strata.items():
            mask = horizon_mask & range_mask
            subset_keys = [k for k, take in zip(keys, mask) if take]
            panel = {"denominators": denominators(subset_keys)}
            for label, p, q in (("tft", z[:, 3], z), ("persistence", baseline, None)):
                per = {}
                for sid in subjects:
                    sm = mask & (ids == sid)
                    per[sid] = reduce_metrics(y[sm], p[sm], q[sm] if q is not None else None)
                    per[sid]["denominators"] = denominators([k for k, take in zip(keys, sm) if take])
                panel[label] = {"micro": reduce_metrics(y[mask], p[mask], q[mask] if q is not None else None),
                                "per_patient": per, "macro_patient": macro(per, subjects)}
                panel[label]["macro_patient"]["denominators"] = panel["denominators"]
            panel["tft_minus_persistence"] = {mode: {key: (
                panel["tft"][mode]["point"][key] - panel["persistence"][mode]["point"][key])
                if panel["tft"][mode]["point"][key] is not None else None
                for key in ("mae", "rmse", "bias", "mard")} for mode in ("micro", "macro_patient")}
            panel["tft_minus_persistence"]["per_patient"] = {
                sid: {key: panel["tft"]["per_patient"][sid]["point"][key] - panel["persistence"]["per_patient"][sid]["point"][key]
                      if panel["tft"]["per_patient"][sid]["N"] else None for key in ("mae", "rmse", "bias", "mard")}
                for sid in subjects}
            ranges[name] = panel
        panels[str(horizon)] = ranges
    return {"revision": REVISION, "interpretation": INTERPRETATION, "units": {"point": "mg/dL", "mard": "%", "probabilities": "proportion"},
            "denominators": denominators(keys), "panels": panels}


def readback(directory):
    directory = Path(directory)
    completion = json.loads((directory/"completion.json").read_text())
    require(sha256(directory/"manifest.json") == completion["manifest_sha256"], "artifact", "readback", "manifest changed")
    manifest = json.loads((directory/"manifest.json").read_text())
    require(manifest["revision"] == REVISION, "compatibility", "readback", "metric revision changed")
    require(set(manifest["artifact_sha256"]) == {"raw.json", "metrics.json"}, "artifact", "readback", "incomplete artifact map")
    require(manifest["status"] in ("VALID", "EMPTY"), "artifact", "readback", "not a completed result")
    for name, wanted in manifest["artifact_sha256"].items():
        require(sha256(directory/name) == wanted, "artifact", "readback", "artifact hash mismatch")
    raw = json.loads((directory/"raw.json").read_text())
    result = report_metrics(raw["keys"], raw["y"], np.asarray(raw["z"]).reshape(-1, 7), raw["persistence"], raw["subjects"])
    require(result == json.loads((directory/"metrics.json").read_text()), "artifact", "readback", "metric replay mismatch")
    require(digest(raw["keys"]) == manifest["ordered_keys_sha256"], "artifact", "readback", "key hash mismatch")
    return result


def evaluate(model, dataset, args, frame, *, context, output_dir):
    require(output_dir is not None, "artifact", "input", "new output directory required")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=False)
    planned = None; flow = None; keys = []
    try:
        validate_context(context, model, dataset, frame)
        selected, windows, keys, flow = qualify(dataset, frame, context.split_role)
        planned = len(keys)
        before = state_hash(model)
        if windows:
            loader = selected.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0, shuffle=False)
            model.eval()
            with torch.no_grad():
                predictions = model.predict(loader, return_x=True, return_y=True, mode="quantiles",
                    trainer_kwargs={"accelerator": "cpu", "logger": False, "enable_progress_bar": False})
            y, z = align(predictions, selected, windows)
            baseline, lineage = persistence(windows, frame)
        else:
            y, z, baseline, lineage = np.empty(0), np.empty((0, 7)), np.empty(0), []
        require(before == state_hash(model), "context", "postpredict", "model state mutated")
        require(all(p.grad is None for p in model.parameters()), "context", "postpredict", "unexpected gradients")
        subjects = sorted(frame.subject_id.astype(str).unique().tolist())
        metrics = report_metrics(keys, y, z, baseline, subjects)
        raw = {"keys": keys, "y": y.tolist(), "z": z.tolist(), "persistence": baseline.tolist(), "lineage": lineage, "subjects": subjects}
        atomic_json(out/"raw.json", raw); atomic_json(out/"metrics.json", metrics)
        manifest = {"revision": REVISION, "status": "VALID" if planned else "EMPTY", "planned_N": planned,
                    "flow": flow, "failures": {}, "ordered_keys_sha256": digest(keys), "source_sha256": context.source_sha256,
                    "model_file_sha256": context.receipt.file_sha256, "model": context.receipt.metadata,
                    "numerical_profile": verify_numerics("cpu"), "split_role": context.split_role,
                    "interpretation": INTERPRETATION, "denominators": metrics["denominators"],
                    "artifact_sha256": {name: sha256(out/name) for name in ("raw.json", "metrics.json")}}
        log.info("Raw within-subject evaluation conditional on eligibility: status=%s, N=%s; clinical reliability NOT EVALUATED", manifest["status"], planned)
        atomic_json(out/"manifest.json", manifest)
        atomic_json(out/"completion.json", {"status": manifest["status"], "manifest_sha256": sha256(out/"manifest.json")})
        return metrics
    except Exception as error:
        # Record the error and re-raise. Never return partial/favorable metrics.
        code = error.code if isinstance(error, EvaluationError) else "finite" if isinstance(error, FloatingPointError) else "execution"
        atomic_json(out/"invalid.json", {"status": "INVALID", "planned_N": planned, "flow": flow,
            "ordered_keys_sha256": digest(keys), "error_code": code,
            "stage": getattr(error, "stage", "predict/execution"), "error_type": type(error).__name__, "failure_counts": {code: 1},
            "interpretation": INTERPRETATION})
        raise
