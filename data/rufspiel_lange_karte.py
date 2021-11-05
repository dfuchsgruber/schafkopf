import numpy as np
import os
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torch.data import Dataset
import torch

SEED_TEST_SPLIT = 1337

class RufspielLangeKarteDataset(Dataset):

    """ Dataset class from a replay buffer. """

    def __init__(self, directory):
        self.directory = directory
        self._files = os.listdir(self.directory)

    def __len__(self):
        return 32 * len(self._files)

    def __getitem__(self, idx):
        data_dict = np.load(os.path.join(self.directory, self._files[idx // 32]))
        return {
            'state' : torch.tensor(data_dict['state'][idx % 32]).long(),
            'action' : torch.tensor(data_dict['action'][idx % 32]).long(),
            'current_trick' : torch.tensor(data_dict['current_trick'][idx % 32]).long(),
            'agent_cards' : torch.tensor(data_dict['agent_cards'][idx % 32]).long(),
            'player' : torch.tensor(data_dict['player'][idx % 32]).long(),
            'reward' : torch.tensor(data['rewards'][idx % 32]).float(),
            'color' : torch.tensor(data['color']).long(),
        }

class RufspielLangeKarteDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int = 128, split: list = [0.8, 0.1, 0.1]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.data = RufspielLangeKarteDataset(self.data_dir)
        sizes = [int(len(self.data) * size) for size in sizes]
        data_train_val, self.data_test = random_split(self.data, [sizes[0] + sizes[1], sizes[2]], generator=torch.Generator().manual_seed(SEED_TEST_SPLIT))
        self.data_train, self.data_val = random_split(data_train_val, [sizes[0], sizes[1]])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

