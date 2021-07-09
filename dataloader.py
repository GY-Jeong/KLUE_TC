import os
import torch
import pandas as pd


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def load_data(self, file_name):
        csv_file_name = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_name)
        return df.values

    def load_train_data(self):
        self.train_data = self.load_data('train_data.csv')

    def load_test_data(self):
        self.test_data = self.load_data('test_data.csv')


class YNAT_dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, is_inference):
        self.args = args
        self.data = data
        self.is_inference = is_inference

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        #print(type(row))
        # np.array -> torch.tensor 형변환
        #for i, col in enumerate(row):
        #    if type(col) == str:
        #        pass
        #    else:
        #        row[i] = torch.tensor(col)

        return row

