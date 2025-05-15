from torch.utils.data import Dataset
from torch import from_numpy
import pandas as pd
import numpy as np

pd.set_option("future.no_silent_downcasting", True)

class TitanicDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = self.extract_data(pd.read_csv(csv_file))

    def extract_data(self, frame: pd.DataFrame) -> pd.DataFrame:
        res = frame[["Pclass", "Sex", "Age", "Survived"]].copy()
        res["Sex"] = res["Sex"].replace(["male", "female"], [0, 1])
        res["Age"] = res["Age"].fillna(float(res["Age"].mode()))
        return res
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        return (from_numpy(self.dataframe.iloc[idx, :-1].to_numpy(dtype=np.float32)), 
                from_numpy(self.dataframe.iloc[idx, -1:].to_numpy(dtype=np.float32)))