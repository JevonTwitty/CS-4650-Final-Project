import pandas as pd

def load_dataset(foldername='data', filename='AI_Detection'):
    file_path = f"{foldername}/{filename}.csv"
    data = pd.read_csv(file_path)
    data.columns = 'text', 'label'

    return data