"""
DATE: 10/09/2021
LAST CHANGE: 29/09/2021
AUTHOR: CHENG ZHANG

load data file and preprocess
"""
import os
import numpy as np
from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
import random
import re
import math


class ReadData(Dataset):
    def __init__(self, cover_dir, stego_dir, list, b_filter):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.list = list
        self.b_filter = b_filter
        self.cover_label = np.array([0], dtype='int32')
        self.stego_label = np.array([1], dtype='int32')
        assert len(self.list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        idx = int(idx)
        (file_tuple, index_tuple) = self.list[idx]
        (start_index, end_index) = index_tuple
        (file, label) = file_tuple
        if label == 1:
            path = self.stego_dir + "/" + file
            label = self.stego_label
        else:
            path = self.cover_dir + "/" + file
            label = self.cover_label

        data = np.loadtxt(path)
        data = data[start_index:end_index,]
        samples = {'data': data, 'label': label}
        return samples


class SpeedTestReadData(Dataset):
    def __init__(self, cover_dir, stego_dir, list, keep):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.list = list
        self.keep = keep
        self.cover_label = np.array([0], dtype='int32')
        self.stego_label = np.array([1], dtype='int32')
        assert len(self.list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        idx = int(idx)
        (file, label) = self.list[idx]
        if label == 1:
            path = self.stego_dir + "/" + file
            label = self.stego_label
        else:
            path = self.cover_dir + "/" + file
            label = self.cover_label

        data = np.loadtxt(path)
        data = data[:self.keep, :]
        samples = {'data': data, 'label': label}
        return samples


class BalancedRandomData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset) * 2

    def __getitem__(self, index):
        reference_index = index // 2
        reference_reminder = index % 2
        current_data = self.dataset[reference_index]['images'][reference_reminder]
        current_label = np.array([self.dataset[reference_index]['labels'][reference_reminder]], dtype='int32')

        samples = {'data': current_data, 'label': current_label}
        return samples


def read_data(cover_dir, stego_dir, batch_size, num_workers, pin_memory, language, sample_length):
    # get file names, corresponding to chinese and english
    # Language: 'English' or 'Chinese'
    dir_cover = cover_dir + "/" + language
    # list = [language + "/" + x.split("\\")[-1] for x in glob(dir_cover + "/" + '*')]

    b_filter = None

    list = [language + "/" + re.split(r"[/,\\]", x)[-1] for x in glob(dir_cover + "/" + '*.*')]
    random.shuffle(list)

    # calculate the length of test, valid, and train set
    len_list = len(list)
    len_train = int(len_list * 0.6)
    len_valid = int(len_list * 0.8)

    # get file names corresponding to test and train set
    list_train = list[:len_train]
    list_valid = list[len_train:len_valid]
    list_test = list[len_valid:]

    # get combined train data and test data file list
    cover_train = [(file, 0) for file in list_train]
    stego_train = [(file, 1) for file in list_train]
    cover_valid = [(file, 0) for file in list_valid]
    stego_valid = [(file, 1) for file in list_valid]
    cover_test = [(file, 0) for file in list_test]
    stego_test = [(file, 1) for file in list_test]
    train_file = cover_train + stego_train
    valid_file = cover_valid + stego_valid
    test_file = cover_test + stego_test


    train_file = add_split(sample_length, train_file)
    test_file = add_split(sample_length, test_file)
    valid_file = add_split(sample_length, valid_file)

    random.shuffle(train_file)
    random.shuffle(test_file)
    random.shuffle(valid_file)

    dataset_train = ReadData(cover_dir, stego_dir, train_file, b_filter)
    dataset_test = ReadData(cover_dir, stego_dir, test_file, b_filter)
    dataset_valid = ReadData(cover_dir, stego_dir, valid_file, b_filter)

    sampler_train = RandomSampler(dataset_train)
    sampler_valid = RandomSampler(dataset_valid)
    sampler_test = RandomSampler(dataset_test)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler_train,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, sampler=sampler_valid,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, sampler=sampler_test,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    return train_loader, valid_loader, test_loader


def add_split(sample_length, file_list):
    temp_file_list = []
    frames = int(int((1000*sample_length)//10))
    to_split = math.floor(10/sample_length)
    for file in file_list:
        for index in range(to_split):
            temp_file_list.append((file, (index*frames, (index+1)*frames)))
    return temp_file_list


def speed_test_read_data(cover_dir, stego_dir, batch_size, num_workers, pin_memory, language, sample_length):
    # get file names, corresponding to chinese and english
    # Language: 'English' or 'Chinese'
    dir_cover = cover_dir + "/" + language
    # list = [language + "/" + x.split("\\")[-1] for x in glob(dir_cover + "/" + '*')]

    list = [language + "/" + re.split(r"[/,\\]", x)[-1] for x in glob(dir_cover + "/" + '*.*')]

    # get combined train data and test data file list
    cover_speed = [(file, 0) for file in list]
    stego_speed = [(file, 1) for file in list]

    random.shuffle(cover_speed)
    random.shuffle(stego_speed)

    cover_speed = cover_speed[:2500]
    stego_speed = stego_speed[:2500]

    speed_file = cover_speed + stego_speed

    keep = int((1000*sample_length)//10)

    dataset_speed = SpeedTestReadData(cover_dir, stego_dir, speed_file, keep)

    sampler_speed = RandomSampler(dataset_speed)

    speed_test_loader = DataLoader(dataset_speed, batch_size=batch_size, sampler=sampler_speed,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    return speed_test_loader


def get_inference_value_read_data(cover_dir, stego_dir, batch_size, num_workers, pin_memory, language, sample_length):
    # get file names, corresponding to chinese and english
    # Language: 'English' or 'Chinese'
    dir_cover = cover_dir + "/" + language
    # list = [language + "/" + x.split("\\")[-1] for x in glob(dir_cover + "/" + '*')]

    b_filter = None

    list = [language + "/" + re.split(r"[/,\\]", x)[-1] for x in glob(dir_cover + "/" + '*.*')]
    random.shuffle(list)

    # get combined train data and test data file list
    cover = [(file, 0) for file in list]
    stego = [(file, 1) for file in list]

    file = cover + stego

    get_inference_value_file = add_split(sample_length, file)

    dataset = ReadData(cover_dir, stego_dir, get_inference_value_file, b_filter)

    sampler_train = RandomSampler(dataset)

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler_train,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    return loader