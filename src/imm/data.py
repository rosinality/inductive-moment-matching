import io
import pickle

from ffrecord import FileReader
import numpy as np
from PIL import Image
import torch
from torch.utils import data


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = image.convert("RGB")

        while min(*image.size) >= 2 * self.size:
            image = image.resize(
                tuple(size // 2 for size in image.size), resample=Image.BOX
            )

        scale = self.size / min(image.size)
        image = image.resize(
            tuple(round(size * scale) for size in image.size), resample=Image.BICUBIC
        )

        image = np.array(image)
        crop_y = (image.shape[0] - self.size) // 2
        crop_x = (image.shape[1] - self.size) // 2
        image = image[crop_y : crop_y + self.size, crop_x : crop_x + self.size]
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image


class ImageDataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.reader = FileReader(path, check_data=False)
        self.transform = transform

    def __len__(self):
        return self.reader.n - 1  # metadata is at -1

    def __getitem__(self, index):
        data = pickle.loads(self.reader.read_one(index))
        image, label = data[2], data[0]
        image = Image.open(io.BytesIO(image))

        if self.transform is not None:
            image = self.transform(image)

        return image, label, index


class LatentDataset(data.Dataset):
    def __init__(self, path):
        self.reader = FileReader(path, check_data=False)

    def __len__(self):
        return self.reader.n

    def __getitem__(self, index):
        latent = self.reader.read_one(index)
        latent: tuple[torch.Tensor, torch.Tensor, int, int] = pickle.loads(
            latent
        )  # mean, std, label, index

        return latent
