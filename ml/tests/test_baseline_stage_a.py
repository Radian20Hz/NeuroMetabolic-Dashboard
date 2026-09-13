"""Stage A: synthetic-only audit; no model fit, patient files or evaluation.

Run from repo root: .venv/bin/python -m unittest discover -s ml/tests -v
Known defects are expectedFailure tests of the desired invariant. An unexpected
success requires review/removal of that marker, not silent acceptance.
"""
import logging
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from ml.scripts import preprocess_ohiot1dm as pre
from ml.scripts import train_tft_population_v2 as pipeline


def write_xml(path, n=480, start="2024-01-01", shift=0, missing=(), future_at=None):
    """Invented signals only; XML schema matches the production parser."""
    root = ET.Element("patient")
    groups = {name: ET.SubElement(root, name) for name in (
        "glucose_level", "bolus", "basal", "meal", "exercise",
        "basis_heart_rate", "basis_gsr", "basis_skin_temperature", "basis_steps",
    )}
    times = pd.date_range(start, periods=n, freq="5min")
    for i, timestamp in enumerate(times):
        ts = timestamp.strftime("%d-%m-%Y %H:%M:%S")
        changed = future_at is not None and i >= future_at
        if i not in missing:
            ET.SubElement(groups["glucose_level"], "event", ts=ts,
                          value=str(110 + shift + i % 43 + (70 if changed else 0)))
        ET.SubElement(groups["basal"], "event", ts=ts, value=str(0.8 + i % 3 / 10))
        if i % 37 == 0:
            ET.SubElement(groups["bolus"], "event", ts_begin=ts, dose="2")
            ET.SubElement(groups["meal"], "event", ts=ts, carbs="20")
        if i % 71 == 0:
            ET.SubElement(groups["exercise"], "event", ts=ts, duration="15")
        for name, value in (("basis_heart_rate", 65 + i % 20),
                            ("basis_gsr", 1 + i % 7 / 10),
                            ("basis_skin_temperature", 32 + i % 4 / 10),
                            ("basis_steps", i % 11)):
            ET.SubElement(groups[name], "event", ts=ts,
                          value=str(value + (5 if changed else 0)))
    ET.ElementTree(root).write(path)
    return times


def processed(path, **kwargs):
    write_xml(path, **kwargs)
    return pre.process_patient(path, "synthetic_A")


class StageA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger().setLevel(logging.ERROR)
        cls.tmp = tempfile.TemporaryDirectory(prefix="nmd-stage-a-")
        cls.root = Path(cls.tmp.name)
        parts = []
        for subject, shift in (("synthetic_A", 0), ("synthetic_B", 80)):
            for split, n, start in (("train", 480, "2024-01-01"),
                                    ("test", 120, "2024-01-02 16:00")):
                path = cls.root / f"{subject}-{split}.xml"
                write_xml(path, n=n, start=start, shift=shift)
                parts.append(pre.process_patient(path, subject, source_split=split))
        cls.source = pd.concat(parts, ignore_index=True)
        cls.source.to_parquet(cls.root / "training.parquet", index=False)
        cls.train, cls.val = pipeline.load_and_preprocess_data(cls.root)
        cls.test = pipeline.load_test_data(
            cls.root, max(cls.train.time_idx.max(), cls.val.time_idx.max()))
        cls.args = SimpleNamespace(context=48, horizon=12)
        cls.training_ds, cls.validation_ds = pipeline.build_datasets(
            cls.train, cls.val, cls.args)
        cls.test_ds = pipeline.create_time_series_dataset(
            cls.test, reference_dataset=cls.training_ds)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_event_binning_and_sensor_matching_are_backward_only(self):
        root = ET.fromstring('''<patient>
          <glucose_level><event ts="01-01-2024 00:00:00" value="100"/>
          <event ts="01-01-2024 00:04:00" value="200"/>
          <event ts="01-01-2024 00:10:00" value="120"/></glucose_level>
          <bolus><event ts_begin="01-01-2024 00:04:00" dose="2"/></bolus>
          <meal><event ts="01-01-2024 00:04:00" carbs="20"/></meal>
          <exercise><event ts="01-01-2024 00:04:00" duration="15"/></exercise>
          <basis_heart_rate><event ts="01-01-2024 00:04:00" value="80"/></basis_heart_rate>
        </patient>''')
        frame = pre.resample_to_cgm_grid(
            pre._parse_glucose(root), pre._parse_bolus(root), pre._parse_basal(root),
            pre._parse_temp_basal(root), pre._parse_meals(root),
            pre._parse_exercise(root), pre._parse_basis_data(root))
        for column, value in (("bolus_event", 2), ("meal_event", 20),
                              ("exercise_duration_min", 15)):
            self.assertEqual(frame[column].iloc[0], 0)
            self.assertEqual(frame[column].iloc[1], value)
        self.assertEqual(frame.glucose_mg_dl.iloc[0], 100)
        self.assertEqual(frame.glucose_mg_dl.iloc[1], 200)
        self.assertTrue(pd.isna(frame.heart_rate.iloc[0]))
        self.assertEqual(frame.heart_rate.iloc[1], 80)

    def test_ffill_uses_past_and_marks_imputed_rows(self):
        frame = processed(self.root / "gap.xml", n=150, missing=range(30, 40))
        indexed = frame.set_index("timestamp")
        times = pd.date_range("2024-01-01", periods=150, freq="5min")
        for i in range(30, 36):
            self.assertEqual(indexed.loc[times[i], "glucose_mg_dl"], 110 + 29)
            self.assertEqual(indexed.loc[times[i], "cgm_gap_flag"], 1)
        self.assertFalse(times[36] in indexed.index)
        self.assertEqual(indexed.loc[times[40], "cgm_gap_flag"], 0)

    def test_active_preprocessing_features_are_prefix_invariant(self):
        original = processed(self.root / "prefix.xml", n=180)
        changed = processed(self.root / "suffix.xml", n=180, future_at=100)
        columns = sorted(set(pipeline.TIME_VARYING_UNKNOWN_REALS) & set(original.columns))
        np.testing.assert_allclose(original.loc[:99, columns], changed.loc[:99, columns],
                                   atol=1e-7, rtol=1e-7, equal_nan=True)

    def test_future_meal_bolus_exercise_do_not_change_past_aggregates(self):
        n = 200
        original = pd.DataFrame({"bolus_event": np.zeros(n), "meal_event": np.zeros(n),
                                 "exercise_duration_min": np.zeros(n)})
        changed = original.copy()
        changed.loc[120:, :] = 5.
        for function in (pre.compute_iob_cob, pre.add_rolling_insulin_carb, pre.add_exercise_features):
            a, b = function(original.copy()), function(changed.copy())
            np.testing.assert_allclose(a.iloc[:120], b.iloc[:120], atol=1e-8)

    def test_metadata_auditor_detects_overlap_and_duplicates(self):
        from ml.tests.audit_stage_a_metadata import summarize
        metadata = self.source[["subject_id", "source_split", "timestamp"]].copy()
        clean = summarize(metadata)
        self.assertEqual(clean["train_test_ordered"], 2)
        self.assertEqual(clean["overlap_rows"], 0)
        contaminated = metadata.iloc[[0]].copy()
        contaminated["source_split"] = "test"
        bad = summarize(pd.concat([metadata, contaminated, contaminated], ignore_index=True))
        self.assertEqual(bad["overlap_rows"], 1)
        self.assertEqual(bad["duplicate_rows"], 1)
        self.assertEqual(bad["train_test_ordered"], 1)

    @unittest.expectedFailure
    def test_A01_trend_feature_must_not_read_future(self):
        original = pd.Series(np.arange(80, dtype=float) + 100)
        changed = original.copy()
        changed.iloc[40:] += 100
        np.testing.assert_allclose(pre._rolling_linear_deviation(original, 12).iloc[:40],
                                   pre._rolling_linear_deviation(changed, 12).iloc[:40],
                                   atol=1e-8)

    @unittest.expectedFailure
    def test_A02_resting_hr_before_first_observation_must_not_read_future(self):
        original = pd.Series([np.nan] * 30 + [70.] * 40)
        changed = pd.Series([np.nan] * 30 + [120.] * 40)
        np.testing.assert_allclose(pipeline._compute_hr_resting_causal(original).iloc[:30],
                                   pipeline._compute_hr_resting_causal(changed).iloc[:30])

    def test_postsplit_features_are_prefix_invariant_with_observed_start(self):
        original = self.source.query("subject_id == 'synthetic_A' and source_split == 'train'").copy()
        changed = original.copy()
        columns = ["glucose_mg_dl", "bolus_event", "meal_event", "heart_rate", "gsr",
                   "skin_temperature", "steps_last_30m", "insulin_on_board", "carb_on_board"]
        changed.loc[changed.index[250:], columns] += 10
        left = pipeline._compute_long_window_features(original).iloc[:250]
        right = pipeline._compute_long_window_features(changed).iloc[:250]
        numeric = left.select_dtypes(include="number").columns
        np.testing.assert_allclose(left[numeric], right[numeric], equal_nan=True)

    @unittest.expectedFailure
    def test_A06_future_sensor_availability_must_not_change_active_past_feature(self):
        original = self.source.query("subject_id == 'synthetic_A' and source_split == 'train'").copy()
        original["skin_temperature"] = np.nan
        changed = original.copy()
        changed.loc[changed.index[300:], "skin_temperature"] = 33.
        left = pipeline._compute_exercise_features_causal(original)
        right = pipeline._compute_exercise_features_causal(changed)
        np.testing.assert_allclose(left.autonomic_stress_index.iloc[:300],
                                   right.autonomic_stress_index.iloc[:300])

    @unittest.expectedFailure
    def test_A07_correction_prior_has_same_definition_in_train_and_validation(self):
        # Same contiguous history, compare actual loader train feature against
        # helper used by validation/test (not against a duplicated formula).
        tr = self.train[self.train.subject_id == "synthetic_A"]
        expected = pipeline._compute_correction_bolus_prior_with_history(tr)
        np.testing.assert_allclose(tr.correction_bolus_prior.values, expected.values)

    def test_known_defects_have_numeric_reproducers(self):
        # Characterization alongside expectedFailure invariants ensures that
        # those markers do not conceal an unrelated exception in a fixture.
        g = pd.Series(np.arange(80, dtype=float) + 100)
        changed = g.copy()
        changed.iloc[40:] += 100
        a, b = [pre._rolling_linear_deviation(v, 12) for v in (g, changed)]
        self.assertGreater(abs(a.iloc[35] - b.iloc[35]), 1)
        self.assertNotIn("glucose_deviation_from_trend", pipeline.TIME_VARYING_UNKNOWN_REALS)
        hr = pipeline._compute_hr_resting_causal(pd.Series([np.nan] * 30 + [123.] * 40))
        self.assertEqual(hr.iloc[0], 123.)
        frame = self.train[self.train.subject_id == "synthetic_A"].copy()
        removed = set(frame.time_idx.iloc[100:124])
        ds = pipeline.create_time_series_dataset(
            frame[~frame.time_idx.isin(removed)], reference_dataset=self.training_ds)
        count = 0
        for i, row in ds.decoded_index.reset_index(drop=True).iterrows():
            start, end = int(row.time_idx_first_prediction), int(row.time_idx_last)
            missing = set(range(start, end + 1)) & removed
            if missing:
                _, (target, weight) = ds[i]
                self.assertIsNone(weight)
                for step in missing:
                    self.assertTrue(torch.isfinite(target[step - start]))
                    # PF fills missing integer timesteps from the preceding row.
                    self.assertAlmostEqual(float(target[step - start]),
                                           float(frame.loc[frame.time_idx == 99, "glucose_mg_dl"].iloc[0]))
                count += 1
        self.assertGreater(count, 0)

    def test_split_membership_chronology_and_test_isolation(self):
        for subject in ("synthetic_A", "synthetic_B"):
            tr, va, te = [df[df.subject_id == subject] for df in (self.train, self.val, self.test)]
            self.assertEqual((len(tr), len(va), len(te)), (408, 72, 120))
            self.assertLess(tr.timestamp.max(), va.timestamp.min())
            self.assertLess(va.timestamp.max(), te.timestamp.min())
            self.assertFalse(set(tr.timestamp) & set(va.timestamp))
            self.assertFalse(set(te.timestamp) & set(pd.concat([tr, va]).timestamp))
        changed = self.source.copy()
        changed.loc[changed.source_split == "test", "glucose_mg_dl"] = 399.
        with patch.object(pipeline.pd, "read_parquet", return_value=changed):
            tr, va = pipeline.load_and_preprocess_data(self.root)
        pd.testing.assert_frame_equal(self.train, tr)
        pd.testing.assert_frame_equal(self.val, va)

    def test_time_idx_preserves_three_hour_gap_within_each_split(self):
        for df in (self.train, self.val, self.test):
            frame = df.copy()
            for subject in frame.subject_id.unique():
                indices = frame.index[frame.subject_id == subject]
                frame.loc[indices[len(indices)//2:], "timestamp"] += pd.Timedelta("3h")
            result = pipeline._assign_gapped_time_idx(frame)
            for _, group in result.groupby("subject_id"):
                expected = group.timestamp.diff().dt.total_seconds().iloc[1:] / 300
                np.testing.assert_array_equal(group.time_idx.diff().iloc[1:], expected)
                self.assertIn(37, expected.values)

    def test_time_idx_rejects_duplicates_reverse_and_offgrid(self):
        for minutes in ([0, 0], [5, 0], [0, 6]):
            with self.subTest(minutes=minutes), self.assertRaises(RuntimeError):
                pipeline._timestamp_steps_5min(pd.Series(
                    pd.Timestamp("2024-01-01") + pd.to_timedelta(minutes, unit="min")))

    def test_split_index_offset_does_not_remove_real_observations(self):
        for subject in self.train.subject_id.unique():
            tr = self.train[self.train.subject_id == subject]
            va = self.val[self.val.subject_id == subject]
            self.assertEqual(va.timestamp.min() - tr.timestamp.max(), pd.Timedelta("5min"))
            self.assertEqual(va.time_idx.min() - tr.time_idx.max(), pipeline.TRAIN_VAL_GAP + 1)

    @unittest.expectedFailure
    def test_A03_one_hour_rolling_must_expire_events_across_long_gap(self):
        frame = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01", "2024-01-01 03:05"], format="mixed"),
                              "bolus_event": [2., 0.], "meal_event": [20., 0.]})
        out = pre.add_rolling_insulin_carb(frame)
        self.assertEqual(out.bolus_last_1h.iloc[-1], 0.)

    @unittest.expectedFailure
    def test_A03_iob_must_expire_after_more_than_kernel_duration(self):
        frame = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01", "2024-01-01 06:00"], format="mixed"),
                              "bolus_event": [2., 0.], "meal_event": [0., 0.]})
        self.assertAlmostEqual(pre.compute_iob_cob(frame).insulin_on_board.iloc[-1], 0.)

    def test_warmstart_rejects_future_or_overlapping_history(self):
        history = self.train[self.train.subject_id == "synthetic_A"]
        for current in (history.tail(10), history.head(10)):
            with self.assertRaises(RuntimeError):
                pipeline._compute_long_window_features(current, pipeline._extract_train_warmstart(history))
            with self.assertRaises(RuntimeError):
                pipeline._compute_correction_bolus_prior_with_history(current, history)

    def test_contiguous_warmstart_matches_past_lags(self):
        for subject in self.train.subject_id.unique():
            tr = self.train[self.train.subject_id == subject]
            va = self.val[self.val.subject_id == subject]
            self.assertEqual(va.glucose_lag_1.iloc[0], tr.glucose_mg_dl.iloc[-1])
            self.assertEqual(va.glucose_lag_24.iloc[0], tr.glucose_mg_dl.iloc[-24])

    def test_dataset_windows_stay_within_subject_and_split_and_targets_align(self):
        for frame, dataset in ((self.train, self.training_ds), (self.val, self.validation_ds),
                               (self.test, self.test_ds)):
            decoded = dataset.decoded_index.reset_index(drop=True)
            self.assertEqual(len(decoded), len(dataset))
            for i, row in decoded.iterrows():
                group = frame[frame.subject_id == row.subject_id].set_index("time_idx")
                indices = range(int(row.time_idx_first), int(row.time_idx_last) + 1)
                self.assertTrue(set(indices).issubset(group.index))
                x, (target, _) = dataset[i]
                start = int(row.time_idx_first_prediction)
                expected = group.loc[list(range(start, int(row.time_idx_last) + 1)), "glucose_mg_dl"]
                np.testing.assert_allclose(target.numpy(), expected.values)
                self.assertEqual(int(x["encoder_length"]), start - int(row.time_idx_first))
                self.assertTrue(torch.isfinite(x["x_cont"]).all())

    def test_validation_is_full_rolling_and_normalizer_is_reused(self):
        self.assertGreater(len(self.validation_ds), self.val.subject_id.nunique())
        for ds in (self.validation_ds, self.test_ds):
            self.assertEqual(ds.min_prediction_length, 12)
            self.assertEqual(ds.max_prediction_length, 12)
            pd.testing.assert_frame_equal(self.training_ds.target_normalizer.norm_, ds.target_normalizer.norm_)
        self.assertIn("glucose_mg_dl", self.training_ds.time_varying_unknown_reals)
        self.assertFalse(set(pipeline.TIME_VARYING_KNOWN_REALS) &
                         {"glucose_mg_dl", "basal_rate", "bolus_last_1h", "carbs_last_1h", "heart_rate"})

    def test_future_decoder_values_do_not_change_encoder_or_known_decoder_inputs(self):
        frame = self.val[self.val.subject_id == "synthetic_A"].copy()
        a = pipeline.create_time_series_dataset(frame, reference_dataset=self.training_ds, predict_mode=True)
        cutoff = int(a.decoded_index.time_idx_first_prediction.iloc[0])
        frame.loc[frame.time_idx >= cutoff, pipeline.TIME_VARYING_UNKNOWN_REALS] += 20.
        b = pipeline.create_time_series_dataset(frame, reference_dataset=self.training_ds, predict_mode=True)
        xa, _ = a[0]
        xb, _ = b[0]
        encoder_length = int(xa["encoder_length"])
        torch.testing.assert_close(xa["x_cont"][:encoder_length], xb["x_cont"][:encoder_length])
        # PF keeps unknown decoder columns in x_cont; TFT must select known
        # decoder variables. This checks dataset inputs, not model execution.
        for name in pipeline.TIME_VARYING_KNOWN_REALS:
            j = a.reals.index(name)
            torch.testing.assert_close(xa["x_cont"][encoder_length:, j], xb["x_cont"][encoder_length:, j])

    @unittest.expectedFailure
    def test_A04_decoder_targets_must_not_be_synthesized_inside_missing_gap(self):
        frame = self.train[self.train.subject_id == "synthetic_A"].copy()
        removed = set(frame.time_idx.iloc[100:124])  # two hours without rows
        frame = frame[~frame.time_idx.isin(removed)]
        ds = pipeline.create_time_series_dataset(frame, reference_dataset=self.training_ds)
        for row in ds.decoded_index.itertuples():
            targets = set(range(int(row.time_idx_first_prediction), int(row.time_idx_last) + 1))
            self.assertFalse(targets & removed, "Dataset contains unobserved decoder targets")

    @unittest.expectedFailure
    def test_A05_imputed_cgm_targets_must_be_excluded_or_zero_weighted(self):
        frame = self.val[self.val.subject_id == "synthetic_A"].copy()
        idx = frame.index[40]
        frame.loc[idx, "glucose_mg_dl"] = frame.loc[idx - 1, "glucose_mg_dl"]
        frame.loc[idx, "cgm_gap_flag"] = 1.
        ds = pipeline.create_time_series_dataset(frame, reference_dataset=self.training_ds)
        for i, row in ds.decoded_index.reset_index(drop=True).iterrows():
            start, end = int(row.time_idx_first_prediction), int(row.time_idx_last)
            if start <= frame.loc[idx, "time_idx"] <= end:
                _, (_, weight) = ds[i]
                self.assertIsNotNone(weight, "Imputed target receives default full weight")
                self.assertEqual(float(weight[int(frame.loc[idx, "time_idx"]) - start]), 0.)


if __name__ == "__main__":
    unittest.main()
