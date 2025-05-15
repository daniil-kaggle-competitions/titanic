from TitanicDataset import TitanicDataset
from TitanicModel import TitanicModel
from torch.utils.data import DataLoader
import pandas as pd
import torch

def main():
    model = TitanicModel()
    model.load_state_dict(torch.load('best.pt', weights_only=True))
    model.eval()

    dataset = TitanicDataset("test.csv", eval=True)
    res = pd.DataFrame(columns=["PassengerId", "Survived"])
    for i, data in enumerate(dataset):
        input, passid = data
        output = model(input)
        survived = 1
        if output < 0.5:
            survived = 0
        
        res.loc[i] = [int(passid), survived]

    res.to_csv('ans.csv', index=False)


if __name__ == "__main__":
    main()