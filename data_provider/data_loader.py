from pathlib import Path

import pandas as pd

# Helper functions
def read_csv(foldername, file_name):
    file_path = foldername + "/" + file_name

    return pd.read_csv(file_path)

def drop_duplicate_texts(df):
    if "text" in df.columns:
        return df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    
    return df.drop_duplicates().reset_index(drop=True)

def clean_text_label_dataframe(dataframe):
    dataframe = dataframe.dropna(subset=["text", "label"]).copy()
    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["label"] = dataframe["label"].astype(int)
    return drop_duplicate_texts(dataframe)

# Dataset-specific loading functions
def load_ai_human_dataset(foldername):
    df = read_csv(foldername, "AI_Human.csv")

    df = df[["text", "generated"]].copy()
    df.columns = ["text", "label"]

    return clean_text_label_dataframe(df)


def load_ai_detection_dataset(foldername):
    df = read_csv(foldername, "AI_Detection.csv")

    return clean_text_label_dataframe(df)


def load_multi_model_detection_dataset_binary(foldername):
    df = read_csv(foldername, "multi_model_detection.csv")

    source_columns = [column for column in df.columns if column not in {"prompt", "Human_story"}]

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "text": row["Human_story"],
            "label": 0,
            "source": "Human_story",
            "prompt": row["prompt"],
        })

        for source_column in source_columns:
            rows.append({
                "text": row[source_column],
                "label": 1,
                "source": source_column,
                "prompt": row["prompt"],
            })

    df = pd.DataFrame(rows)

    return clean_text_label_dataframe(df)


def load_train_v2_dataset(foldername):
    df = read_csv(foldername, "train_v2_drcat_02.csv")
    df = df[["text", "label"]].copy()

    return clean_text_label_dataframe(df)

# Route standard dataset loading through main function
def load_standard_dataset(foldername, file_name):

    if file_name == "ai_human.csv":
        return load_ai_human_dataset(foldername)

    if file_name == "ai_detection.csv":
        return load_ai_detection_dataset(foldername)

    if file_name == "multi_model_detection.csv":
        return load_multi_model_detection_dataset_binary(foldername)

    if file_name == "train_v2_drcat_02.csv":
        return load_train_v2_dataset(foldername)

    df = read_csv(foldername, file_name)

    if {"text", "label"}.issubset(df.columns):
        df = df[["text", "label"]].copy()

        if pd.api.types.is_numeric_dtype(df["label"]):
            df["label"] = df["label"].astype(int)

        return clean_text_label_dataframe(df)

    df = df.iloc[:, :2].copy()
    df.columns = ["text", "label"]
    if pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(int)
        
    return clean_text_label_dataframe(df)

# Experiment loading functions
def load_experiment1(foldername, split=None):
    train_df = pd.concat(
        [load_ai_human_dataset(foldername), load_multi_model_detection_dataset_binary(foldername)],
        ignore_index=True,
    )
    train_df = clean_text_label_dataframe(train_df)

    test_df = load_ai_detection_dataset(foldername)

    if split is None:
        return {"train": train_df, "test": test_df}

    if split == "train":
        return train_df

    if split == "test":
        return test_df 

def load_experiment2(foldername):
    merged_df = pd.concat(
        [
            load_ai_human_dataset(foldername),
            load_ai_detection_dataset(foldername),
            load_multi_model_detection_dataset_binary(foldername),
        ],
        ignore_index=True,
    )

    return clean_text_label_dataframe(merged_df)


def load_experiment3(foldername, split=None):
    train_df = pd.concat(
        [
            load_ai_detection_dataset(foldername),
            load_ai_human_dataset(foldername),
            load_train_v2_dataset(foldername),
        ],
        ignore_index=True,
    )
    train_df = clean_text_label_dataframe(train_df)

    test_df = load_multi_model_detection_dataset_binary(foldername)

    if split is None:
        return {"train": train_df, "test": test_df}

    if split == "train":
        return train_df

    if split == "test":
        return test_df

# Main loading function
def load_dataset(foldername="data", file_name="AI_Detection.csv", split=None):
    """
    Load a dataset or experiment bundle by file name.

    Supported file names:
    - AI_Human.csv: returns a df with columns [text, label]
    - AI_Detection.csv: returns a df with columns [text, label]
    - multi_model_detection.csv: returns a binary df with columns [text, label, source, prompt]
    - train_v2_drcat_02.csv: returns a df with columns [text, label]

    Supported experiments:
    - experiment1:
      train = AI_Human + multi_model_detection
      test = AI_Detection
      If split is None, returns {"train": ..., "test": ...}
      If split is 'train' or 'test', returns just that df.
    - experiment2:
      returns one df combining AI_Human + AI_Detection + multi_model_detection
    - experiment3:
      train = AI_Detection + AI_Human + train_v2_drcat_02
      test = multi_model_detection
      If split is None, returns {"train": ..., "test": ...}
      If split is 'train' or 'test', returns just that df.

    Note: to select an experiment, set file_name to the experiment name (e.g. "experiment1"). 
    For the standard datasets, set file_name to the csv file name ("AI_Human.csv or AI_Human").

    All final dfs are deduplicated on the text column when present.
    """

    normalized_name = file_name.lower().removesuffix(".csv")

    if normalized_name == "experiment1":
        return load_experiment1(foldername, split=split)

    if normalized_name == "experiment2":
        return load_experiment2(foldername)

    if normalized_name == "experiment3":
        return load_experiment3(foldername, split=split)

    return load_standard_dataset(foldername, file_name)
