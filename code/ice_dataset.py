import os
import os.path as path
import json
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class ICEDataset(data.Dataset):
    def __init__(self, data_dir="data/train"):
        super(ICEDataset, self).__init__()

        image_names = [
            path.join(data_dir, f[:-9])
            for f in os.listdir(data_dir)
            if f.endswith("mask.npy")
        ]
        image_name2data_files = {}
        for image_name in image_names:
            data_files = {}
            data_files["image_file"] = image_name + ".jpg"
            data_files["mask_file"] = image_name + "_mask.npy"
            data_files["label_file"] = image_name + "_label.json"
            data_files["projected_file"] = image_name + \
                "_segmentation_projected.npy"
            data_files["outlined_file"] = image_name + \
                "_segmentation_outlined.npy"
            data_files["reconstructed_file"] = image_name + \
                "_segmentation_reconstructed.npy"
            image_name2data_files[image_name] = data_files

        self.image_names_data_files = image_name2data_files.items()

    def __len__(self):
        return len(self.image_names_data_files)

    def __getitem__(self, index):
        image_name, data_files = self.image_names_data_files[index]

        image = np.array(Image.open(data_files["image_file"]))
        mask = np.load(data_files["mask_file"]).astype(np.float32)
        with open(data_files["label_file"]) as f:
            label = json.load(f)
        hardness = np.array(label["hardness"])
        organs = np.array(label["organs"])
        segmentation_projected = np.load(
            data_files["projected_file"]
        ).astype(np.float32)
        segmentation_reconstructed = np.load(
            data_files["reconstructed_file"]
        ).astype(np.float32)
        segmentation_outlined = np.load(
            data_files["outlined_file"]
        ).astype(np.float32)

        image = image.transpose(2, 0, 1)
        image = image / 255.0
        image = image * 2.0 - 1.0
        segmentation_projected = segmentation_projected * 2.0 - 1.0
        segmentation_outlined = segmentation_outlined * 2.0 - 1.0
        segmentation_reconstructed = segmentation_reconstructed * 2 - 1.0

        return {
            "image_name": image_name,
            "image": torch.FloatTensor(image),
            "hardness": torch.LongTensor(hardness),
            "organs": torch.LongTensor(organs),
            "segmentation_projected": torch.FloatTensor(
                segmentation_projected
            ),
            "segmentation_reconstructed": torch.FloatTensor(
                segmentation_reconstructed
            ),
            "segmentation_outlined": torch.FloatTensor(segmentation_outlined)
        }
