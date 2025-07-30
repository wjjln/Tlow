import gin
import os
import random
import torch
import sys
from tokenizer.schemas import SeqBatch
from enum import Enum
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional
import numpy as np
import pickle as pkl

class ItemData(Dataset):
    def __init__(
        self,
        dataset_folder,
        dim=512,
        **kwargs
    ) -> None:
        
        item_data = []
        for folder in dataset_folder:
            if '.pt' in folder:
                item_emb = torch.load(folder, weights_only=False)
                item_emb = item_emb[0]['item']['x']
            elif '.feat' in folder or '.sent_emb' in folder:
                item_emb = np.fromfile(folder, dtype=np.float32).reshape(-1, dim)
                item_emb = torch.from_numpy(item_emb).float()
            elif '.npy' in folder:
                item_emb = np.load(folder)
                item_emb = torch.from_numpy(item_emb).float()
            elif '.pkl' in folder:
                item_emb = pkl.load(open(folder, "rb"))
                item_emb = torch.from_numpy(item_emb).float()
            else:
                raise ValueError(f"Unsupported file type in {folder}. Supported types are .pt, .feat, .sent_emb, .npy, and .pkl.")
            item_data.append(item_emb)
        self.item_data = torch.cat(item_data, dim=0)
        print('item_data shape:', self.item_data.shape)

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        item_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        # x = self.item_data[idx, :768]
        x = self.item_data[idx]
        return SeqBatch(
            user_ids=-1 * torch.ones_like(item_ids.squeeze(0)),
            ids=item_ids,
            ids_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            x=x,
            x_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            seq_mask=torch.ones_like(item_ids, dtype=bool)
        )