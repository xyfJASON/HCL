import os
from PIL import Image

import torch
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


def split_coco(p):
    filename = os.path.splitext(os.path.basename(p))[0]
    filename = filename.split('-')
    return filename[0]


class LOGO_30K(Dataset):
    def __init__(self, root, split='train', transform=None, mask_transform=None):
        root = os.path.join(root, '27kpng')
        assert os.path.isdir(root), f'{root} is not an existing directory'
        assert split in ['train', 'valid']
        img_root = os.path.join(root, 'train_images' if split == 'train' else 'val_images')
        assert os.path.isdir(img_root), f'{img_root} is not a valid directory'

        self.transform = transform
        self.mask_transform = mask_transform

        self.img_paths = extract_images(os.path.join(root, 'natural'))
        self.corrupted_paths = extract_images(os.path.join(img_root, 'image'))
        if split != 'train':
            # delete extra 4 images in ./val_images/image/
            self.corrupted_paths = sorted(list(set(self.corrupted_paths) - {
                os.path.join(img_root, 'image', 'COCO_val2014_000000201186-Burger_King_Logo-177_wm.png'),
                os.path.join(img_root, 'image', 'COCO_val2014_000000201186-Burger_King_Logo-177_mask.png'),
                os.path.join(img_root, 'image', 'COCO_val2014_000000201186-Burger_King_Logo-177_input.png'),
                os.path.join(img_root, 'image', 'COCO_val2014_000000201186-Burger_King_Logo-177_coarse.png'),
            }))
        self.mask_paths = extract_images(os.path.join(img_root, 'mask'))
        self.wm_paths = extract_images(os.path.join(img_root, 'wm'))

        corrupted_ids = [split_coco(p) for p in self.corrupted_paths]
        mask_ids = [split_coco(p) for p in self.mask_paths]
        wm_ids = [split_coco(p) for p in self.wm_paths]
        assert corrupted_ids == mask_ids == wm_ids

        self.img_paths = [os.path.join(root, 'natural', i + '.jpg') for i in corrupted_ids]
        assert len(self.img_paths) == len(self.corrupted_paths) == len(self.mask_paths) == len(self.wm_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.corrupted_paths[item]).convert('RGB')
        gt_img = Image.open(self.img_paths[item]).convert('RGB')
        wm = Image.open(self.wm_paths[item]).convert('RGB')
        mask = Image.open(self.mask_paths[item])
        if self.transform is not None:
            X = self.transform(X)
            gt_img = self.transform(gt_img)
            wm = self.transform(wm)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return X, gt_img, wm, mask


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
    dataset = LOGO_30K(root='/data/xuyifeng/data/LOGO/', split='train')
    print(len(dataset))
    dataset = LOGO_30K(root='/data/xuyifeng/data/LOGO/', split='valid')
    print(len(dataset))


if __name__ == '__main__':
    _test()
