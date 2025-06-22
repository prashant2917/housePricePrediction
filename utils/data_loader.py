import  pandas as pd
def load_data_from_csv(path)->pd.DataFrame:
    return pd.read_csv(path)