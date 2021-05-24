import os

import numpy as np
import pandas as pd
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from torch.utils import data

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def image_loader(img_path, accimage=False):
    """Loads an image.
    Args:
        img_path (str): path to image file
        accimage (bool): whether to use accimage loader. `True` means that it would be used.
    """
    if accimage:
        import accimage
        img = accimage.Image(img_path)
    else:
        with Image.open(img_path) as img:
            img = img.convert('RGB')

    # XXX added by Yagi
    trans_func = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return trans_func(img)


class EvalDataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data_file, root_dir="", accimage=False, transform=None, header=True, index=False):
        """Initialization"""
        self.transform = transform
        self.accimage = accimage
        self.header = header
        self.root_dir = root_dir
        with open(data_file) as f:
            lines = f.readlines()
            file_name, ext = os.path.splitext(data_file)
            if ext == '.txt':
                self.data_list = [line.strip('\n') for line in lines]
            else:
                self.data_list = [line.strip('\n').split(',')[0 + int(index)] for line in lines]
                self.label_list = [line.strip('\n').split(',')[1 + int(index)] for line in lines]
                if header:
                    self.data_list = self.data_list[1:]
                    self.label_list = self.label_list[1:]

                self.label_list = [int(x) for x in self.label_list]

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.data_list)

    def __getitem__(self, index):
        """Generates one sample of data"""
        img_path = os.path.join(self.root_dir, self.data_list[index])
        try:
            img_data = image_loader(img_path, accimage=self.accimage)
            if self.transform is not None:
                img_data = self.transform(img_data)
            return index, img_data, self.label_list[index]
        except FileNotFoundError:
            print(f'File not found {img_path}')
        except Exception as e:
            print(f'Error loading image: {img_path}. Error: {e}')
        return None


class ModifiedImageFolder(datasets.DatasetFolder):
    """Provides additional functionality to report images which could not be loaded"""

    def __init__(self, root, out_dir, transform=None, target_transform=None,
                 loader=image_loader, is_valid_file=None, stage=None):
        super(ModifiedImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                  transform=transform,
                                                  target_transform=target_transform,
                                                  is_valid_file=is_valid_file)

        self.error_files = list()
        self.stage = stage
        self.out_dir = out_dir

    def __getitem__(self, index):
        """Returns the datapoint (and its class information) at a particular index
        Args:
            index (int): Index of the data point in the dataset
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        try:
            sample = self.loader(path)
        except:
            self.error_files.append(path)
            pd.DataFrame(self.error_files).to_csv(f'{self.out_dir}/error_files_{self.stage}.csv')
            return self.__getitem__(index + 1)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target