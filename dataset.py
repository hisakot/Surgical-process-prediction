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
        process = torch.tensor(process, dtype=torch.float32)

        return area, label

def make_dataset():
    dataset_dicts = list()
    area = np.loadtxt(common.AREA_CSV, delimiter=",", skiprows=1, usecols=(1, 2, 3))

    process = np.loadtxt(common.PROCESS_CSV, delimiter=",")

    print(area.shape, process.shape)
    exit()

    for i, org_path in enumerate(org_paths):
        if gaze_points[i][0] == 0 and gaze_points[i][1] == 0:
            continue
#         hand_path = org_path.replace("org_imgs", "contour")
#         tool_path = org_path.replace("org_imgs", "tool_masks")
#         cutting_path = org_path.replace("org_imgs", "cutting_area")
        hand_path = org_path.replace("org_imgs", "flow_hand")
        tool_path = org_path.replace("org_imgs", "flow_tool")
        cutting_path = org_path.replace("org_imgs", "flow_cut")
        dataset_dicts.append({"org_path" : org_path,
                              "hand_path" : hand_path,
                              "tool_path" : tool_path,
                              "cutting_path" : cutting_path,
                              "flow_path" : flow_path,
                              "gaze_point" : gaze_points[i],})

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
