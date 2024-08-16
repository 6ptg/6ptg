import gzip
import numpy as np
import torch
from torch.utils.data import Dataset
import os, glob
from torchvision.io import read_image
from tqdm import tqdm
from model.tools import get_file_path
from model import bit_transform
import multiprocessing
import gc

class GeneralDataset(Dataset):
    def __init__(self, x_data, y_data):

        self.x_data = x_data.clone().detach()
        self.y_data = y_data.clone().detach()
        self.len = self.x_data.shape[0]
        self.channel = self.x_data.shape[1]
        self.mean = torch.zeros(self.channel)
        self.std = torch.ones(self.channel)
        self.num_class = 0


    def get_normalize(self):
        for i in range(self.channel):
            self.mean[i] = torch.mean(self.x_data[:, i, :, :])
            self.std[i] = torch.std(self.x_data[:, i, :, :])

    def __getitem__(self, item):
        x_data = self.x_data[item]
        y_data = self.y_data[item]

        for i in range(self.channel):
            x_data[i, :, :] -= self.mean[i]
            x_data[i, :, :] /= self.std[i]

        return x_data, y_data

    def __len__(self):
        return self.len

class IPv6SuffixDataset(GeneralDataset):
    def __init__(self, file_path, dataset_mean=torch.zeros(3), dataset_std=torch.ones(3), noise_size=1, num_workers=12):
        self.num_workers = num_workers
        self.noise_size = noise_size
        ipv6_address_list = []

        length = 0
        with open(file_path, 'r') as file_object:
            for content in file_object:
                length += 1

        pool = multiprocessing.Pool(self.num_workers)
        mp_res = []

        pbar = tqdm(total=length)
        pbar.set_description("Dataset is reading")
        with open(file_path, 'r') as file_object:
            for content in file_object:
                pbar.update()
                if len(content) < 5:
                    continue
                ipv6_address_list.append(content.replace('\n', ''))
        pbar.close()

        pbar = tqdm(total=length)
        pbar.set_description("Dataset is constructing")
        update = lambda *args: pbar.update()

        for ipv6_address in ipv6_address_list:
            mp_res.append(pool.apply_async(bit_transform.single_bit_transform_suffix, (ipv6_address, self.noise_size,), callback=update))

        pool.close()
        pool.join()
        pbar.close()

        prefix_list = []
        suffix_list = []

        for res in mp_res:
            prefix, suffix = res.get()
            prefix_list.append(prefix)
            suffix_list.append(suffix)

        self.x_data = torch.tensor(suffix_list, dtype=torch.float32)
        self.y_data = torch.tensor(prefix_list, dtype=torch.float32)

        super().__init__(self.x_data, self.y_data)

        self.mean = dataset_mean
        self.std = dataset_std



class MnistDataset(GeneralDataset):
    def __init__(self, x_data, y_data, num_class=10, channel=3):
        super().__init__(x_data, y_data)
        self.num_class = num_class
        self.x_data = self.x_data.expand([-1, channel, self.x_data.shape[2], self.x_data.shape[3]])/255.0
        self.channel = self.x_data.shape[1]
        self.mean = torch.zeros(self.channel)
        self.std = torch.ones(self.channel)
        self.get_normalize()

        self.y_data = torch.zeros(self.x_data.shape[0], self.num_class).scatter_(1, self.y_data, 1)

class TinyDataset(GeneralDataset):
    def __init__(self, x_data, y_data, num_class=200, channel=3):
        super().__init__(x_data, y_data)
        self.num_class = num_class
        self.x_data = self.x_data / 255.0
        self.channel = self.x_data.shape[1]
        self.mean = torch.zeros(self.channel)
        self.std = torch.ones(self.channel)
        self.get_normalize()

        self.y_data = torch.zeros(self.x_data.shape[0], self.num_class).scatter_(1, self.y_data, 1)

class IPv6PrefixDataset(GeneralDataset):
    def __init__(self, file_path, dataset_mean=torch.zeros(3), dataset_std=torch.ones(3), noise_size=1, num_workers=12):
        self.num_workers = num_workers
        self.noise_size = noise_size
        ipv6_address_list = []

        length = 0
        with open(file_path, 'r') as file_object:
            for content in file_object:
                length += 1

        pool = multiprocessing.Pool(self.num_workers)
        mp_res = []

        pbar = tqdm(total=length)
        pbar.set_description("Dataset is reading")
        with open(file_path, 'r') as file_object:
            for content in file_object:
                pbar.update()
                if len(content) < 5:
                    continue
                ipv6_address_list.append(content.replace('\n', ''))
        pbar.close()

        pbar = tqdm(total=length)
        pbar.set_description("Dataset is constructing")
        update = lambda *args: pbar.update()

        for ipv6_address in ipv6_address_list:
            mp_res.append(pool.apply_async(bit_transform.single_bit_transform_prefix, (ipv6_address, self.noise_size,), callback=update))

        pool.close()
        pool.join()
        pbar.close()

        prefix_list = []
        suffix_list = []

        for res in mp_res:
            prefix, suffix = res.get()
            prefix_list.append(prefix)
            suffix_list.append(suffix)

        self.x_data = torch.tensor(prefix_list, dtype=torch.float32)
        self.y_data = torch.tensor(suffix_list, dtype=torch.float32)

        super().__init__(self.x_data, self.y_data)

        self.mean = dataset_mean
        self.std = dataset_std


class IPv6MixDatasetFromIP(GeneralDataset):
    def __init__(self, file_path, prefix_mean=torch.zeros(3), prefix_std=torch.ones(3), suffix_mean=torch.zeros(3), suffix_std=torch.ones(3), noise_size=1, num_workers=12):
        self.num_workers = num_workers
        self.noise_size = noise_size
        ipv6_address_list = []

        length = 0
        with open(file_path, 'r') as file_object:
            for content in file_object:
                length += 1

        pool = multiprocessing.Pool(self.num_workers)
        mp_res = []

        pbar = tqdm(total=length)
        pbar.set_description("Dataset is reading")
        with open(file_path, 'r') as file_object:
            for content in file_object:
                pbar.update()
                if len(content) < 5:
                    continue
                ipv6_address_list.append(content.replace('\n', ''))
        pbar.close()

        pbar = tqdm(total=length)
        pbar.set_description("Dataset is constructing")
        update = lambda *args: pbar.update()

        for ipv6_address in ipv6_address_list:
            mp_res.append(pool.apply_async(bit_transform.single_bit_transform_mix, (ipv6_address, self.noise_size,), callback=update))

        pool.close()
        pool.join()
        pbar.close()

        prefix_list = []
        suffix_list = []

        for res in mp_res:
            prefix, suffix = res.get()
            prefix_list.append(prefix)
            suffix_list.append(suffix)

        self.x_data = torch.tensor(prefix_list, dtype=torch.float32)
        self.y_data = torch.tensor(suffix_list, dtype=torch.float32)

        super().__init__(self.x_data, self.y_data)
        self.prefix_mean = prefix_mean
        self.prefix_std = prefix_std
        self.suffix_mean = suffix_mean
        self.suffix_std = suffix_std

    def __getitem__(self, item):
        x_data = self.x_data[item]
        y_data = self.y_data[item]

        for i in range(self.channel):
            x_data[i, :, :] -= self.prefix_mean[i]
            x_data[i, :, :] /= self.prefix_std[i]

        for i in range(self.channel):
            y_data[i, :, :] -= self.suffix_mean[i]
            y_data[i, :, :] /= self.suffix_std[i]

        return x_data, y_data

class IPv6MixDataset(GeneralDataset):
    def __init__(self, file_path, prefix_mean=torch.zeros(3), prefix_std=torch.ones(3), suffix_mean=torch.zeros(3), suffix_std=torch.ones(3), noise_size=1, num_workers=12):
        self.num_workers = num_workers
        self.noise_size = noise_size

        self.x_data = torch.load(f"{file_path}/prefix.pt")
        self.y_data = torch.load(f"{file_path}/suffix.pt")

        self.len = self.x_data.shape[0]
        self.channel = self.x_data.shape[1]

        self.prefix_mean = prefix_mean
        self.prefix_std = prefix_std
        self.suffix_mean = suffix_mean
        self.suffix_std = suffix_std

    def __getitem__(self, item):
        x_data = self.x_data[item]
        y_data = self.y_data[item]

        for i in range(self.channel):
            x_data[i, :, :] -= self.prefix_mean[i]
            x_data[i, :, :] /= self.prefix_std[i]

        for i in range(self.channel):
            y_data[i, :, :] -= self.suffix_mean[i]
            y_data[i, :, :] /= self.suffix_std[i]

        return x_data, y_data


if __name__ == "__main__":
    pass


