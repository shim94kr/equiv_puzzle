import numpy as np
import torch
from torch.utils.data import Dataset
from dnnlib import EasyDict as edict
from glob import glob

def npz_to_dict(data):
    data_dict = edict()
    for key in data.files:
        data_dict[key] = data[key]
    return data_dict

def pad_for_batch(data, n_base):
    for k, v in data.items():
        n = n_base * 2 if k == "rels" else n_base
        v = v[:n] if n - v.shape[0] < 0 else v
        padding = np.zeros((n-v.shape[0],) + v.shape[1:], dtype=np.int64)
        v = np.concatenate((v, padding), 0)

        if k == "piece_mask" or k == "global_mask" or k == "faces_mask" or k == "rels_mask":
            padding = np.zeros(v.shape[:1] + (n-v.shape[1],) + v.shape[2:], dtype=np.int64)
            v = np.concatenate((v, padding), 1)

        data[k] = np.float32(v)
    return data
    

class CrosscutDataset(Dataset):
    def __init__(self, 
                 exp_name,
                 split, 
                 max_num_points = 100):
        super().__init__()
        base_dir = f'./datasets/cross_cut/processed_{split}'
        self.exp_name = exp_name
        self.samples = glob(base_dir + '/*')
        self.n = max_num_points

    def make_encoding(self, data):
        get_one_hot = lambda x, z: np.eye(z)[x]

        data.piece_encoding = np.array([get_one_hot(x, 32) for x in data["piece_idx"]])
        data.corner_encoding = np.array([get_one_hot(x, 32) for x in data["corner_idx"]])
        data.padding_mask = np.ones([len(data["vertices"])])
        data.piece_mask = np.matmul(data.piece_encoding, data.piece_encoding.T)
        data.global_mask = np.matmul(np.ones(len(data.piece_encoding))[:, None], np.ones(len(data.piece_encoding))[None])
        data.rels_mask = np.zeros_like(data.global_mask)
        data.rels_mask[data.rels[:, 0], data.rels[:, 1]] = 1.

        return data

    def __getitem__(self, idx):
        with np.load(self.samples[idx], allow_pickle=True) as data:
            data = npz_to_dict(data)
            data = self.make_encoding(data)
            data = pad_for_batch(data, self.n)

        return data
    
    def __len__(self):
        return len(self.samples)

class CrosscutDataset3D(Dataset):
    def __init__(self, 
                 exp_name,
                 split, 
                 max_num_points = 150):
        super().__init__()
        base_dir = f'./datasets/randomcut_3d/processed_{split}'
        self.exp_name = exp_name
        self.samples = glob(base_dir + '/*')
        self.n = max_num_points

    def make_encoding(self, data):
        get_one_hot = lambda x, z: np.eye(z)[x]

        data.piece_encoding = np.array([get_one_hot(x, 32) for x in data["piece_idx"]])
        data.corner_encoding = np.array([get_one_hot(x, 32) for x in data["corner_idx"]])
        data.padding_mask = np.ones([len(data["vertices"])])
        data.faces_padding_mask = np.ones([len(data["faces_mask"])])
        data.piece_mask = np.matmul(data.piece_encoding, data.piece_encoding.T)
        data.global_mask = np.matmul(np.ones(len(data.piece_encoding))[:, None], np.ones(len(data.piece_encoding))[None])
        data.rels_mask = np.zeros_like(data.global_mask)
        data.rels_mask[data.rels[:, 0], data.rels[:, 1]] = 1.

        return data
    
    def __getitem__(self, idx):
        with np.load(self.samples[idx], allow_pickle=True) as data:
            data = npz_to_dict(data)
            data = self.make_encoding(data)
            data = pad_for_batch(data, self.n)
        return data
    
    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    dataset = CrosscutDataset('puzzlefusion', 'test')
    dataloader_kwargs = edict(pin_memory=True, num_workers=4, prefetch_factor=2)
    dataloader = iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=512, **dataloader_kwargs))
    data = next(dataloader)
    # dict_keys(['piece_idx', 'corner_idx', 'vertices', 't', 'rot', 'rels', 'piece_index', 'corner_index', 'padding_mask', 'piece_mask', 'global_mask'])
    print(data.keys())