
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
from scipy.stats import skew, kurtosis

from spiral_model import extract_features


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="NeuroScope",
    page_icon="🧠",
    layout="wide"
)


# -------------------- STYLE --------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}
[data-testid="stHeader"] {
    background-color: #0b1120;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3, p, label, li {
    color: #f1f5f9 !important;
}
.subtitle {
    color: #94a3b8 !important;
    font-size: 17px;
}
.result-card {
    background-color: #172033;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 12px;
}
.result-label {
    color: #94a3b8 !important;
    font-size: 15px;
}
.result-value {
    color: #60a5fa !important;
    font-size: 25px;
    font-weight: bold;
    margin-top: 8px;
}
.warning-card {
    background-color: #422006;
    border: 1px solid #a16207;
    border-radius: 10px;
    padding: 16px;
    color: #fef3c7 !important;
}
.stButton > button {
    background-color: #2563eb;
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: bold;
}
[data-testid="stFileUploader"] {
    background-color: #172033;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


# -------------------- HEADER --------------------

st.title("🧠 NeuroScope")

st.markdown(
    '<p class="subtitle">'
    'Explainable AI for Movement-Based Research Screening'
    '</p>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="warning-card">
<strong>Research prototype only:</strong>
The MSA-versus-PD model is experimental and has not been
clinically validated. Its outputs are not medical diagnoses.
Voice and spiral assessments are separate Parkinson's-versus-
healthy-control experiments; they do not detect MSA.
</div>
""", unsafe_allow_html=True)


# -------------------- GYROSCOPE FEATURES --------------------

CHANNELS = [
    "gyroThumbX",
    "gyroThumbY",
    "gyroThumbZ",
    "gyroIndexX",
    "gyroIndexY",
    "gyroIndexZ"
]

STAT_NAMES = [
    "mean",
    "std",
    "median",
    "min",
    "max",
    "range",
    "mean_abs",
    "rms",
    "iqr",
    "skew",
    "kurtosis",
    "diff_std",
    "diff_mean_abs",
    "zero_crossings"
]


def extract_gyro_features(mat_path):
    data = scipy.io.loadmat(mat_path)
    features = {}

    for channel in CHANNELS:
        if channel not in data:
            raise ValueError(
                f"Required gyroscope channel missing: {channel}"
            )

        signal = np.asarray(
            data[channel], dtype=float
        ).squeeze()

        signal = signal[np.isfinite(signal)]

        if signal.size < 2:
            raise ValueError(
                f"Not enough valid readings in {channel}."
            )

        differences = np.diff(signal)
        q25, q75 = np.percentile(signal, [25, 75])

        values = [
            np.mean(signal),
            np.std(signal, ddof=1),
            np.median(signal),
            np.min(signal),
            np.max(signal),
            np.max(signal) - np.min(signal),
            np.mean(np.abs(signal)),
            np.sqrt(np.mean(signal ** 2)),
            q75 - q25,
            skew(signal, bias=True),
            kurtosis(signal, bias=True),
            np.std(differences, ddof=1),
            np.mean(np.abs(differences)),
            np.mean(
                np.signbit(signal[1:])
                != np.signbit(signal[:-1])
            )
        ]

        for name, value in zip(STAT_NAMES, values):
            features[f"{channel}_{name}"] = value

    feature_frame = pd.DataFrame([features])
    feature_frame = feature_frame.replace(
        [np.inf, -np.inf], np.nan
    )

    return feature_frame


# -------------------- MODEL LOADING --------------------

@st.cache_resource
def load_msa_model():
    model_path = BASE_DIR / "msa_pd_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "msa_pd_model.pkl was not found in the project folder."
        )

    model_data = joblib.load(model_path)

    if isinstance(model_data, dict):
        if "model" not in model_data:
            raise ValueError(
                "The model file does not contain a 'model' entry."
            )

        return model_data["model"]

    return model_data


@st.cache_resource
def load_voice_model():
    model_data = joblib.load(BASE_DIR / "voice_model.pkl")

    if not isinstance(model_data, dict):
        raise ValueError("Unexpected voice model file format.")

    return model_data["model"], model_data["features"]


@st.cache_resource
def load_spiral_model():
    model_data = joblib.load(BASE_DIR / "spiral_model.pkl")

    if not isinstance(model_data, dict):
        raise ValueError("Unexpected spiral model file format.")

    return model_data["model"], model_data["features"]


# -------------------- PRIMARY MSA / PD ASSESSMENT --------------------

st.header("1. MSA vs Parkinson's Disease")

st.write(
    "Upload a compatible finger-tapping gyroscope recording "
    "in MATLAB .mat format."
)

st.caption(
    "Uses six gyroscope channels from the thumb and index finger. "
    "This is an experimental research demonstration."
)

msa_file = st.file_uploader(
    "Upload finger-tapping .mat recording",
    type=["mat"],
    key="msa_pd_upload"
)

if st.button(
    "Analyze finger-tapping recording",
    type="primary",
    use_container_width=True
):
    if msa_file is None:
        st.error("Please upload a .mat recording first.")
    else:
        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mat",
                delete=False
            ) as temp_file:
                temp_file.write(msa_file.getvalue())
                temporary_path = temp_file.name

            features = extract_gyro_features(temporary_path)
            model = load_msa_model()

            expected_features = None

            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            elif hasattr(model, "named_steps"):
                classifier = model.named_steps.get("classifier")
                if classifier is not None and hasattr(
                    classifier, "feature_names_in_"
                ):
                    expected_features = list(
                        classifier.feature_names_in_
                    )

            if expected_features is not None:
                missing = [
                    name for name in expected_features
                    if name not in features.columns
                ]

                if missing:
                    raise ValueError(
                        "The model expects different features. "
                        "Missing: " + ", ".join(missing[:10])
                    )

                features = features[expected_features]

            prediction = model.predict(features)[0]

            st.divider()
            st.subheader("Screening output")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Model output</div>
                        <div class="result-value">{prediction}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <div class="result-card">
                        <div class="result-label">Input status</div>
                        <div class="result-value">Processed</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]

                if isinstance(joblib.load(
                    BASE_DIR / "msa_pd_model.pkl"
                ), dict):
                    model_data = joblib.load(
                        BASE_DIR / "msa_pd_model.pkl"
                    )
                    classes = model_data.get(
                        "classes", list(model.classes_)
                    )
                else:
                    classes = list(model.classes_)

                score_df = pd.DataFrame({
                    "Class": [str(c) for c in classes],
                    "Model score": probabilities
                })

                st.subheader("Model output scores")
                st.dataframe(
                    score_df.style.format(
                        {"Model score": "{:.2%}"}
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "These scores are not calibrated medical probabilities."
                )

            with st.expander("View extracted gyroscope features"):
                st.dataframe(
                    features.T.rename(
                        columns={0: "Extracted value"}
                    ),
                    use_container_width=True
                )

            st.markdown("""
            <div class="warning-card">
            This experimental output cannot confirm or rule out
            MSA or Parkinson's disease. Do not use it to make
            treatment or healthcare decisions.
            </div>
            """, unsafe_allow_html=True)

        except Exception as error:
            st.error(f"Could not analyze this recording: {error}")
            st.info(
                "Check that the recording contains all six expected "
                "gyroscope channels and uses compatible features."
            )

        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)


# -------------------- EXPLAINABLE AI --------------------

st.divider()
st.header("2. Explainable AI")

st.write(
    "These plots summarize which features influenced the model "
    "during patient-separated cross-validation."
)

explanation_dir = BASE_DIR / "explainability_cv_results"

importance_image = (
    explanation_dir / "cv_shap_feature_importance.png"
)
summary_image = explanation_dir / "cv_shap_summary.png"
importance_csv = (
    explanation_dir / "cv_shap_feature_importance.csv"
)

if importance_image.exists():
    st.subheader("Most influential features")
    st.image(
        str(importance_image),
        use_column_width=True
    )
else:
    st.info(
        "Feature-importance plot not found. Run "
        "explain_msa_pd_cv.py to generate it."
    )

if summary_image.exists():
    st.subheader("SHAP summary plot")
    st.image(
        str(summary_image),
        use_column_width=True
    )

if importance_csv.exists():
    with st.expander("View feature-importance table"):
        importance_data = pd.read_csv(importance_csv)
        st.dataframe(
            importance_data.head(20),
            use_container_width=True,
            hide_index=True
        )

st.caption(
    "SHAP describes model behavior. Feature importance does not "
    "prove medical causation or establish a validated biomarker. "
    "These plots summarize cross-validation models; they are not "
    "individual explanations for the uploaded recording."
)


# -------------------- OPTIONAL VOICE ASSESSMENT --------------------

st.divider()
st.header("3. Optional experimental modules")

st.write(
    "These modules use separate Parkinson's-versus-healthy-control "
    "models. They are not part of the MSA-versus-PD model."
)

with st.expander("Voice assessment"):
    st.write("Upload a UCI voice feature CSV or DATA file.")

    voice_file = st.file_uploader(
        "Choose voice feature file",
        type=["csv", "data"],
        key="voice_upload"
    )

    if st.button("Analyze voice sample", key="analyze_voice"):
        if voice_file is None:
            st.error("Please upload a voice feature file.")
        else:
            try:
                voice_model, voice_features = load_voice_model()
                voice_df = pd.read_csv(voice_file)

                for column in ["name", "status"]:
                    if column in voice_df.columns:
                        voice_df = voice_df.drop(
                            columns=[column]
                        )

                voice_df = voice_df.select_dtypes(
                    include=[np.number]
                )

                missing = [
                    feature for feature in voice_features
                    if feature not in voice_df.columns
                ]

                if missing:
                    st.error(
                        "Missing required features: "
                        + ", ".join(missing)
                    )
                elif voice_df.empty:
                    st.error("No numeric voice features were found.")
                else:
                    sample = voice_df[
                        list(voice_features)
                    ].iloc[[-1]]

                    probabilities = voice_model.predict_proba(sample)[0]
                    classes = list(voice_model.classes_)
                    predicted_class = classes[
                        int(np.argmax(probabilities))
                    ]

                    st.write(
                        "Voice model output:",
                        "Parkinson's pattern"
                        if predicted_class == 1
                        else "Healthy-control pattern"
                    )

                    st.write(
                        "Model output scores:",
                        {
                            str(c): float(p)
                            for c, p in zip(classes, probabilities)
                        }
                    )

                    st.caption(
                        "Experimental Parkinson's-versus-control result "
                        "only; not an MSA assessment."
                    )

            except Exception as error:
                st.error(f"Voice assessment failed: {error}")


# -------------------- OPTIONAL SPIRAL ASSESSMENT --------------------

with st.expander("Spiral drawing assessment"):
    st.write("Upload a spiral recording TXT file.")

    spiral_file = st.file_uploader(
        "Choose spiral recording",
        type=["txt"],
        key="spiral_upload"
    )

    if st.button("Analyze spiral sample", key="analyze_spiral"):
        if spiral_file is None:
            st.error("Please upload a spiral TXT file.")
        else:
            temporary_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=".txt",
                    delete=False
                ) as temp_file:
                    temp_file.write(spiral_file.getvalue())
                    temporary_path = temp_file.name

                spiral_model, spiral_features = load_spiral_model()

                extracted = extract_features(temporary_path)
                sample = pd.DataFrame([extracted])

                missing = [
                    feature for feature in spiral_features
                    if feature not in sample.columns
                ]

                if missing:
                    st.error(
                        "Missing spiral features: "
                        + ", ".join(missing)
                    )
                else:
                    sample = sample[list(spiral_features)]
                    probabilities = spiral_model.predict_proba(sample)[0]
                    classes = list(spiral_model.classes_)
                    predicted_class = classes[
                        int(np.argmax(probabilities))
                    ]

                    st.write(
                        "Spiral model output:",
                        "Parkinson's pattern"
                        if predicted_class == 1
                        else "Healthy-control pattern"
                    )

                    st.write(
                        "Model output scores:",
                        {
                            str(c): float(p)
                            for c, p in zip(classes, probabilities)
                        }
                    )

                    st.caption(
                        "Experimental Parkinson's-versus-control result "
                        "only; not an MSA assessment."
                    )

            except Exception as error:
                st.error(f"Spiral assessment failed: {error}")

            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.remove(temporary_path)


# -------------------- PROJECT INFORMATION --------------------

st.divider()
st.header("Project information")

st.markdown("""
- **Primary method:** Finger-tapping gyroscope analysis.
- **Main comparison:** MSA versus Parkinson's disease.
- **Feature extraction:** Statistical summaries from six gyroscope channels.
- **Classifier:** Logistic Regression.
- **Explainability:** SHAP feature-importance analysis.
- **Evaluation:** Patient-separated cross-validation.
- **Optional experiments:** Voice and spiral Parkinson's-versus-control models.
""")

st.markdown("""
<div class="warning-card">
<strong>Important limitation:</strong> The current dataset contains
only a small number of patients. Independent external validation
and clinical evaluation are required before any medical use.
</div>
""", unsafe_allow_html=True)