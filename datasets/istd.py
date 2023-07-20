import os
from PIL import Image

import torch
from torch import Tensor
import torchvision.transforms as T
from torch.utils.data import Dataset


def extract_images(root):
    """ Extract all images under root """
    img_ext = ['.png', '.jpg', '.jpeg']
    root = os.path.expanduser(root)
    img_paths = []
    for curdir, subdirs, files in os.walk(root):
        for file in files:
            if os.path.splitext(file)[1].lower() in img_ext:
                img_paths.append(os.path.join(curdir, file))
    img_paths = sorted(img_paths)
    return img_paths


class ISTD(Dataset):
    def __init__(self, root, split='train', transform=None, mask_transform=None):
        assert os.path.isdir(root), f'{root} is not an existing directory'
        assert split in ['train', 'test']
        self.transform = transform
        self.mask_transform = mask_transform
        self.corrupted_paths = extract_images(os.path.join(root, split, split + '_A'))
        self.mask_paths = extract_images(os.path.join(root, split, split + '_B'))
        self.img_paths = extract_images(os.path.join(root, split, split + '_C'))
        assert len(self.img_paths) == len(self.corrupted_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.corrupted_paths[item]).convert('RGB')
        gt_img = Image.open(self.img_paths[item]).convert('RGB')
        mask = Image.open(self.mask_paths[item])
        if self.transform is not None:
            X = self.transform(X)
            gt_img = self.transform(gt_img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        if isinstance(X, Tensor) and isinstance(mask, Tensor):
            shadow = X * (1 - mask)
        else:
            shadow = Image.composite(X, Image.new('L', size=X.size, color=0), mask)
        return X, gt_img, shadow, mask


def get_default_transforms(img_size: int):
    transforms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    mask_transforms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Lambda(lambda m: torch.where(torch.lt(m, 0.5), 1., 0.))
    ])
    return transforms, mask_transforms


def _test():
    dataset = ISTD(root='/Users/jason/data/ISTD/', split='train')
    print(len(dataset))
    dataset = ISTD(root='/Users/jason/data/ISTD/', split='valid')
    print(len(dataset))
    dataset = ISTD(root='/Users/jason/data/ISTD/', split='test')
    print(len(dataset))


if __name__ == '__main__':
    _test()
