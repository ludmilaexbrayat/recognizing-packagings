import torch
import os
from torch.utils.data.dataset import Dataset
from PIL import Image


class MultiLabelDataset(Dataset):
    def __init__(self, df, root_dir_img, transform=None):
        """
        Args:
            df(pandas dataframe):           Target dataframe with class one hot encoded and index name of the images.
            root_dir_img (string):          Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.root_dir_img = root_dir_img
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir_img, str(self.df.index[idx]) + '.jpg')
        img = Image.open(img_name)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = torch.Tensor(self.df.iloc[idx].tolist())

        return img, label
