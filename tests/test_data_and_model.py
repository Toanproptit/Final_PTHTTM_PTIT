import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from diabetes_demo.data import encode_features, load_dataset, make_preprocessor, split_dataset
from diabetes_demo.schema import BINARY_FEATURES, DEMO_PATIENT, FEATURES, NUMERIC_FEATURES, PatientInput
from diabetes_demo.agents import RuleBasedAgent

@pytest.fixture(scope="module")
def prepared_data():
    frame, stats = load_dataset()
    splits = split_dataset(frame)
    return frame, stats, splits

def test_splits_have_no_shared_feature_groups(prepared_data):
    frame, stats, splits = prepared_data
    assert stats["raw_rows"] == 100000
    assert "clinical_notes" not in FEATURES
    assert "target" not in FEATURES
    groups = {name: set(frame.iloc[indices].group) for name, indices in splits.items()}
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    assert not groups["validation"] & groups["train"]
    assert sum(len(indices) for indices in splits.values()) == len(frame)
    assert all(frame.iloc[indices].target.nunique() == 2 for indices in splits.values())

def test_preprocessing_statistics_are_fit_only_on_train(prepared_data):
    frame, _, splits = prepared_data
    X = encode_features(frame)
    pipeline = make_preprocessor()
    pipeline.fit(X.iloc[splits["train"]])
    means = pipeline.named_transformers_["numeric"].named_steps["scaler"].mean_.copy()
    np.testing.assert_allclose(means, X.iloc[splits["train"]][NUMERIC_FEATURES + BINARY_FEATURES].mean().to_numpy())
    pipeline.transform(X.iloc[splits["test"]])
    np.testing.assert_array_equal(means, pipeline.named_transformers_["numeric"].named_steps["scaler"].mean_)
    assert pipeline.transform(X.iloc[splits["test"]]).shape[1] == 15
    no_info = pd.DataFrame([{**DEMO_PATIENT, "smoking_history": "No Info"}])
    never = pd.DataFrame([{**DEMO_PATIENT, "smoking_history": "never"}])
    assert not np.array_equal(pipeline.transform(no_info), pipeline.transform(never))

def test_unknown_category_is_rejected_instead_of_silently_imputed():
    frame = pd.DataFrame([{**DEMO_PATIENT, "gender": "unknown"}])
    with pytest.raises(ValueError):
        encode_features(frame)

def test_missing_and_non_boolean_fields_are_rejected():
    with pytest.raises(ValidationError):
        PatientInput(age=45, gender="Male")
    with pytest.raises(ValidationError):
        PatientInput(**{**DEMO_PATIENT, "hypertension": "false"})
    with pytest.raises(ValidationError):
        PatientInput(**{**DEMO_PATIENT, "age": "45"})

def test_rule_skips_unsupported_range_and_does_not_diagnose_from_flags():
    agent = RuleBasedAgent()
    outside = agent.run(PatientInput(**{**DEMO_PATIENT, "age": 100.0}))
    assert not outside["within_training_age"]
    all_yes = agent.run(PatientInput(**{**DEMO_PATIENT, **{name: True for name in BINARY_FEATURES}}))
    assert all_yes["within_training_age"]
    assert "predicted_dataset_label" not in all_yes
    glucose_outside = agent.run(PatientInput(**{**DEMO_PATIENT, "blood_glucose_level": 500.0}))
    assert not glucose_outside["within_training_domain"]
