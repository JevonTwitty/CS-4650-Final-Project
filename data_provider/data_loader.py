import pandas as pd

def load_dataset(foldername='data', filename='AI_Detection'):
    """
    This is a little scuffed.
    If any dataset doesn't start with 2 columns in the order (text, label), this will break.
    """
    file_path = f"{foldername}/{filename}.csv"
    data = pd.read_csv(file_path)

    data = data.iloc[:, :2]  # Keep only the first two columns
    data.columns = 'text', 'label'

    return data