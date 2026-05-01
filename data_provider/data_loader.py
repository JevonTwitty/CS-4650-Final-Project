from pathlib import Path
import re

import pandas as pd


STANDARD_COLUMNS = [
    "text",
    "binary_label",
    "source_model",
    "dataset_name",
    "domain",
    "original_label",
]


SUPPORTED_DATASETS = {
    "ai_vs_human_text": {
        "filenames": ["ai_vs_human_text.csv", "AI_Human.csv"],
    },
    "daigt_v2_train": {
        "filenames": ["daigt_v2_train.csv", "train_v2_drcat_02.csv"],
    },
    "ai_text_detection_dataset": {
        "filenames": [
            "ai_text_detection_dataset.csv",
            "AI_Detection.csv",
            "AI_Detection(1).csv",
        ],
    },
    "multi_model_detection": {
        "filenames": [
            "multi_model_detection.csv",
            "benchmark_ai_detection_multimodel_2026.csv",
        ],
    },
}


DATASET_ALIASES = {
    "ai_vs_human_text": "ai_vs_human_text",
    "ai_human": "ai_vs_human_text",
    "daigt_v2_train": "daigt_v2_train",
    "train_v2_drcat_02": "daigt_v2_train",
    "ai_text_detection_dataset": "ai_text_detection_dataset",
    "ai_detection": "ai_text_detection_dataset",
    "multi_model_detection": "multi_model_detection",
    "benchmark_ai_detection_multimodel_2026": "multi_model_detection",
}


def clean_text(text):
    """
    Basic text cleaning for all datasets.
    Keeps the text readable while removing extra whitespace.
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_dataset(foldername="data", filename=None):
    """
    Load one supported dataset or all supported datasets from data/raw/.

    Parameters
    ----------
    foldername : str
        Usually "data". If a "raw" subfolder exists, files are loaded from there.
    filename : str or None
        Dataset name, alias, or CSV filename. If None, all supported datasets
        that are present are loaded and concatenated.
    """
    data_dir = _resolve_data_dir(foldername)

    if filename is None:
        frames = []
        for dataset_name in SUPPORTED_DATASETS:
            file_path = _find_dataset_file(data_dir, dataset_name)
            if file_path is not None:
                frames.append(_load_one_dataset(file_path, dataset_name))

        if not frames:
            raise FileNotFoundError(
                f"No supported datasets were found in '{data_dir}'."
            )

        combined_df = pd.concat(frames, ignore_index=True)
        return _finalize_standardized_df(combined_df)

    dataset_name = _resolve_dataset_name(filename)
    file_path = _find_dataset_file(data_dir, dataset_name)

    if file_path is None:
        raise FileNotFoundError(
            f"Could not find a file for dataset '{filename}' in '{data_dir}'."
        )

    df = _load_one_dataset(file_path, dataset_name)
    return _finalize_standardized_df(df)


def make_binary_dataset(df):
    """
    Keep only rows with binary labels "human" or "ai".
    """
    _check_standard_columns(df)

    binary_df = df[df["binary_label"].isin(["human", "ai"])].copy()
    return _finalize_standardized_df(binary_df)


def make_multiclass_dataset(df, min_examples_per_class=100):
    """
    Build a multiclass dataset using source_model as the class label.

    Human text uses the class name "human".
    AI rows with no known generator name are excluded.
    """
    _check_standard_columns(df)

    multiclass_df = df.copy()
    multiclass_df["source_model"] = (
        multiclass_df["source_model"].fillna("").astype(str).str.strip()
    )

    multiclass_df = multiclass_df[
        multiclass_df["source_model"].ne("")
        & multiclass_df["source_model"].ne("unknown_ai")
    ].copy()

    class_counts = multiclass_df["source_model"].value_counts()
    keep_classes = class_counts[class_counts >= min_examples_per_class].index

    multiclass_df = multiclass_df[
        multiclass_df["source_model"].isin(keep_classes)
    ].copy()

    return _finalize_standardized_df(multiclass_df)


def _resolve_data_dir(foldername):
    folder = Path(foldername)

    if folder.name.lower() == "raw" and folder.exists():
        return folder

    raw_folder = folder / "raw"
    if raw_folder.exists():
        return raw_folder

    return folder


def _resolve_dataset_name(filename):
    name = Path(str(filename)).stem.lower()
    if name in DATASET_ALIASES:
        return DATASET_ALIASES[name]

    if name in SUPPORTED_DATASETS:
        return name

    raise ValueError(f"Unsupported dataset name: {filename}")


def _find_dataset_file(data_dir, dataset_name):
    for candidate_name in SUPPORTED_DATASETS[dataset_name]["filenames"]:
        candidate_path = Path(data_dir) / candidate_name
        if candidate_path.exists():
            return candidate_path

    return None


def _load_one_dataset(file_path, dataset_name):
    raw_df = pd.read_csv(file_path, low_memory=False)

    if dataset_name == "ai_vs_human_text":
        return _load_ai_vs_human_text(raw_df)
    if dataset_name == "daigt_v2_train":
        return _load_daigt_v2_train(raw_df)
    if dataset_name == "ai_text_detection_dataset":
        return _load_ai_text_detection_dataset(raw_df)
    if dataset_name == "multi_model_detection":
        return _load_multi_model_detection(raw_df)

    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def _load_ai_vs_human_text(raw_df):
    binary_labels = raw_df["generated"].map(_normalize_binary_label)

    df = pd.DataFrame(
        {
            "text": raw_df["text"],
            "binary_label": binary_labels,
            "source_model": binary_labels.map(_binary_to_source_model),
            "dataset_name": "ai_vs_human_text",
            "domain": "unknown",
            "original_label": raw_df["generated"],
        }
    )
    return _finalize_standardized_df(df)


def _load_daigt_v2_train(raw_df):
    binary_labels = raw_df["label"].map(_normalize_binary_label)

    source_models = []
    for binary_label, raw_source in zip(binary_labels, raw_df["source"]):
        if binary_label == "human":
            source_models.append("human")
        else:
            source_models.append(_clean_source_model(raw_source) or "unknown_ai")

    df = pd.DataFrame(
        {
            "text": raw_df["text"],
            "binary_label": binary_labels,
            "source_model": source_models,
            "dataset_name": "daigt_v2_train",
            "domain": raw_df["prompt_name"].fillna("unknown"),
            "original_label": raw_df["label"],
        }
    )
    return _finalize_standardized_df(df)


def _load_ai_text_detection_dataset(raw_df):
    binary_labels = raw_df["label"].map(_normalize_binary_label)

    df = pd.DataFrame(
        {
            "text": raw_df["text"],
            "binary_label": binary_labels,
            "source_model": binary_labels.map(_binary_to_source_model),
            "dataset_name": "ai_text_detection_dataset",
            "domain": "unknown",
            "original_label": raw_df["label"],
        }
    )
    return _finalize_standardized_df(df)


def _load_multi_model_detection(raw_df):
    binary_labels = raw_df["is_ai_generated"].map(_normalize_binary_label)

    source_models = []
    for binary_label, raw_source in zip(binary_labels, raw_df["source_model"]):
        if binary_label == "human":
            source_models.append("human")
        else:
            source_models.append(_clean_source_model(raw_source) or "unknown_ai")

    df = pd.DataFrame(
        {
            "text": raw_df["text_content"],
            "binary_label": binary_labels,
            "source_model": source_models,
            "dataset_name": "multi_model_detection",
            "domain": raw_df["domain_context"].fillna("unknown"),
            "original_label": raw_df["is_ai_generated"],
        }
    )
    return _finalize_standardized_df(df)


def _normalize_binary_label(value):
    if pd.isna(value):
        return None

    label = str(value).strip().lower()

    if label in {"0", "0.0", "false", "human", "real"}:
        return "human"
    if label in {"1", "1.0", "true", "ai", "generated"}:
        return "ai"

    try:
        numeric_value = float(label)
        return "ai" if numeric_value >= 1 else "human"
    except ValueError:
        pass

    if "human" in label or "real" in label:
        return "human"
    if any(token in label for token in ["ai", "gpt", "llama", "claude", "gemini", "mistral"]):
        return "ai"

    return None


def _binary_to_source_model(binary_label):
    if binary_label == "human":
        return "human"
    if binary_label == "ai":
        return "unknown_ai"
    return None


def _clean_source_model(value):
    if pd.isna(value):
        return None

    value = str(value).strip()
    return value if value else None


def _check_standard_columns(df):
    missing_columns = [column for column in STANDARD_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "DataFrame is missing standardized columns: "
            + ", ".join(missing_columns)
        )


def _finalize_standardized_df(df):
    standardized_df = df.copy()

    for column in STANDARD_COLUMNS:
        if column not in standardized_df.columns:
            standardized_df[column] = None

    standardized_df = standardized_df[STANDARD_COLUMNS]
    standardized_df["text"] = standardized_df["text"].map(clean_text)
    standardized_df["binary_label"] = standardized_df["binary_label"].astype("string")
    standardized_df["source_model"] = standardized_df["source_model"].astype("string")
    standardized_df["dataset_name"] = standardized_df["dataset_name"].astype("string")
    standardized_df["domain"] = standardized_df["domain"].fillna("unknown").astype("string")
    standardized_df["original_label"] = standardized_df["original_label"].astype("string")

    standardized_df = standardized_df[standardized_df["text"] != ""]
    standardized_df = standardized_df[
        standardized_df["binary_label"].isin(["human", "ai"])
    ]
    standardized_df = standardized_df.drop_duplicates(subset="text")
    standardized_df = standardized_df.reset_index(drop=True)

    return standardized_df
