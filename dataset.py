import csv
import cv2
import glob
import numpy as np
import os

import torch
from torch.utils.data import Dataset

import common

class Datas():
    def __init__(self):
        self.dataset = list()
        self.length = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # load area csv
        area = self.dataset[idx]["area"]
        area = torch.tensor(area, dtype=torch.float32)

        # load label
        process = self.dataset[idx]["process"] # (x, y)
        process = torch.tensor(process, dtype=torch.long)

        return area, label

def make_dataset():
    dataset_dicts = list()
    area = np.loadtxt(common.AREA_CSV, delimiter=",", usecols=(1, 2, 3))

    with open(common.PROCESS_CSV, "r") as f:
        reader = csv.reader(f, delimiter=",")
        processes = [row for row in reader]

    for i, process in enumerate(processes):
        dataset_dicts.append({"area" : area[i], "process" : process})

    return dataset_dicts

def setup_data():
    datas = Datas()

    try:
        cache = torch.load(common.DATASET_CACHE)
        datas.dataset = cache["dataset"]
        datas.length = cache["length"]

    except FileNotFoundError:
        dataset_dicts = make_dataset()
        datas.dataset = dataset_dicts
        datas.length = len(datas.dataset)
        print(datas.length)

        cache_dict = {"dataset" : datas.dataset,
                      "length" : datas.length,}
        torch.save(cache_dict, common.DATASET_CACHE)

    return datas
