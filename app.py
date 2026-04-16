from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_rf_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoders.pkl"


@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with ENCODER_PATH.open("rb") as encoder_file:
        encoder_payload = pickle.load(encoder_file)
    return model, encoder_payload


def render_numeric_input(feature_name: str, feature_meta: dict):
    if feature_meta["dtype"] == "int":
        return st.number_input(
            feature_name,
            min_value=int(feature_meta["min"]),
            max_value=int(feature_meta["max"]),
            value=int(feature_meta["default"]),
            step=1,
        )
    return st.number_input(
        feature_name,
        min_value=float(feature_meta["min"]),
        max_value=float(feature_meta["max"]),
        value=float(feature_meta["default"]),
        step=0.1,
        format="%.4f",
    )


def main() -> None:
    st.set_page_config(page_title="UrbanNest Rent Predictor", layout="wide")
    st.title("UrbanNest Analytics")
    st.subheader("Dynamic House Rent Prediction Engine")
    st.write("Provide all property details below to estimate the expected monthly rent.")

    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        st.error(
            "Missing model artifacts. Run `train.ipynb` first to generate the saved files."
        )
        st.stop()

    model, encoder_payload = load_artifacts()
    ui_metadata = encoder_payload["ui_metadata"]
    label_encoders = encoder_payload["label_encoders"]

    feature_columns = ui_metadata["feature_columns"]
    categorical_columns = ui_metadata["categorical_columns"]
    category_options = ui_metadata["category_options"]
    numeric_features = ui_metadata["numeric_features"]

    inputs: dict[str, object] = {}
    left_column, right_column = st.columns(2)

    for index, feature_name in enumerate(feature_columns):
        current_column = left_column if index % 2 == 0 else right_column
        with current_column:
            if feature_name in categorical_columns:
                inputs[feature_name] = st.selectbox(feature_name, category_options[feature_name])
            else:
                inputs[feature_name] = render_numeric_input(feature_name, numeric_features[feature_name])

    if st.button("Predict"):
        inference_row = pd.DataFrame([inputs], columns=feature_columns)
        for column in categorical_columns:
            inference_row[column] = label_encoders[column].transform(inference_row[column].astype(str))
        prediction = float(model.predict(inference_row)[0])
        st.success(f"Predicted monthly rent: INR {prediction:,.2f}")
        st.write(
            f"Best optimization strategy: `{encoder_payload['overall_best_method']}` | "
            f"Final test MAE: `{encoder_payload['test_mae']:.2f}`"
        )


if __name__ == "__main__":
    main()
