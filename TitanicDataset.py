from torch.utils.data import Dataset
from torch import from_numpy
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

pd.set_option("future.no_silent_downcasting", True)

class TitanicDataset(Dataset):
    def __init__(self, csv_file, eval=False):
        self.eval = eval
        self.data = self.extract_data(pd.read_csv(csv_file))

    def extract_data(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.eval:
            res = frame[["Pclass", "Sex", "Age", "PassengerId"]].copy()
        else:
            res = frame[["Pclass", "Sex", "Age", "Survived"]].copy()
        res["Sex"] = res["Sex"].replace(["male", "female"], [0, 1])

        imputer = KNNImputer(n_neighbors=5)
        return imputer.fit_transform(res).astype(dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (from_numpy(self.data[idx, :-1]), 
                from_numpy(self.data[idx, -1:]))