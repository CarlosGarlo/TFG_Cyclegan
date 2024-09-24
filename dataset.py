from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


# class HorseZebraDataset(Dataset):
#     def __init__(self, root_zebra, root_horse, transform=None):
#         self.root_zebra = root_zebra
#         self.root_horse = root_horse
#         self.transform = transform
#
#         self.zebra_images = os.listdir(root_zebra)
#         self.horse_images = os.listdir(root_horse)
#         self.length_dataset = max(len(self.zebra_images), len(self.horse_images))  # 1000, 1500
#         self.zebra_len = len(self.zebra_images)
#         self.horse_len = len(self.horse_images)
#
#     def __len__(self):
#         return self.length_dataset
#
#     def __getitem__(self, index):
#         zebra_img = self.zebra_images[index % self.zebra_len]
#         horse_img = self.horse_images[index % self.horse_len]
#
#         zebra_path = os.path.join(self.root_zebra, zebra_img)
#         horse_path = os.path.join(self.root_horse, horse_img)
#
#         zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
#         horse_img = np.array(Image.open(horse_path).convert("RGB"))
#
#         if self.transform:
#             augmentations = self.transform(image=zebra_img, image0=horse_img)
#             zebra_img = augmentations["image"]
#             horse_img = augmentations["image0"]
#
#         return zebra_img, horse_img


# class SunnyNightDataset(Dataset):
#     def __init__(self, root_night, root_sunny, transform=None):
#         self.root_night = root_night
#         self.root_sunny = root_sunny
#         self.transform = transform
#
#         self.night_images = os.listdir(root_night)
#         self.sunny_images = os.listdir(root_sunny)
#         self.length_dataset = max(len(self.night_images), len(self.sunny_images))  # 1000, 1500
#         self.night_len = len(self.night_images)
#         self.sunny_len = len(self.sunny_images)
#
#     def __len__(self):
#         return self.length_dataset
#
#     def __getitem__(self, index):
#         night_img = self.night_images[index % self.night_len] #renombrar y devolverla a train
#         sunny_img = self.sunny_images[index % self.sunny_len]
#
#         night_path = os.path.join(self.root_night, night_img)
#         sunny_path = os.path.join(self.root_sunny, sunny_img)
#
#         night_img = np.array(Image.open(night_path).convert("RGB"))
#         sunny_img = np.array(Image.open(sunny_path).convert("RGB"))
#
#         if self.transform:
#             augmentations = self.transform(image=night_img, image0=sunny_img)
#             night_img = augmentations["image"]
#             sunny_img = augmentations["image0"]
#
#         return night_img, sunny_img
class SunnyNightDataset(Dataset):
    def __init__(self, root_night, root_sunny, transform=None):
        self.root_night = root_night
        self.root_sunny = root_sunny
        self.transform = transform

        self.night_images = os.listdir(root_night)
        self.sunny_images = os.listdir(root_sunny)
        self.length_dataset = max(len(self.night_images), len(self.sunny_images))  # 1000, 1500
        self.night_len = len(self.night_images)
        self.sunny_len = len(self.sunny_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        night_img_name = self.night_images[index % self.night_len] #renombrar y devolverla a train
        sunny_img_name = self.sunny_images[index % self.sunny_len]

        night_path = os.path.join(self.root_night, night_img_name)
        sunny_path = os.path.join(self.root_sunny, sunny_img_name)

        night_img = np.array(Image.open(night_path).convert("RGB"))
        sunny_img = np.array(Image.open(sunny_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=night_img, image0=sunny_img)
            night_img = augmentations["image"]
            sunny_img = augmentations["image0"]

        return night_img, sunny_img, night_img_name, sunny_img_name

class SunnyCloudyDataset(Dataset):
    def __init__(self, root_cloudy, root_sunny, transform=None):
        self.root_cloudy = root_cloudy
        self.root_sunny = root_sunny
        self.transform = transform

        self.cloudy_images = os.listdir(root_cloudy)
        self.sunny_images = os.listdir(root_sunny)
        self.length_dataset = max(len(self.cloudy_images), len(self.sunny_images))  # 1000, 1500
        self.cloudy_len = len(self.cloudy_images)
        self.sunny_len = len(self.sunny_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        cloudy_img_name = self.cloudy_images[index % self.cloudy_len]
        sunny_img_name = self.sunny_images[index % self.sunny_len]

        cloudy_path = os.path.join(self.root_cloudy, cloudy_img_name)
        sunny_path = os.path.join(self.root_sunny, sunny_img_name)

        cloudy_img = np.array(Image.open(cloudy_path).convert("RGB"))
        sunny_img = np.array(Image.open(sunny_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=cloudy_img, image0=sunny_img)
            cloudy_img = augmentations["image"]
            sunny_img = augmentations["image0"]

        return cloudy_img, sunny_img, cloudy_img_name, sunny_img_name


class NightCloudyDataset(Dataset):
    def __init__(self, root_cloudy, root_night, transform=None):
        self.root_cloudy = root_cloudy
        self.root_night = root_night
        self.transform = transform

        self.cloudy_images = os.listdir(root_cloudy)
        self.night_images = os.listdir(root_night)
        self.length_dataset = max(len(self.cloudy_images), len(self.night_images))  # 1000, 1500
        self.cloudy_len = len(self.cloudy_images)
        self.night_len = len(self.night_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        cloudy_img_name = self.cloudy_images[index % self.cloudy_len]
        night_img_name = self.night_images[index % self.night_len]

        cloudy_path = os.path.join(self.root_cloudy, cloudy_img_name)
        night_path = os.path.join(self.root_night, night_img_name)

        cloudy_img = np.array(Image.open(cloudy_path).convert("RGB"))
        night_img = np.array(Image.open(night_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=cloudy_img, image0=night_img)
            cloudy_img = augmentations["image"]
            night_img = augmentations["image0"]

        return cloudy_img, night_img, cloudy_img_name, night_img_name
