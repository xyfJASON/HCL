import os
from PIL import Image

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


class Raindrop(Dataset):
    def __init__(self, root, split='train', transform=None):
        assert os.path.isdir(root), f'{root} is not an existing directory'
        assert split in ['train', 'test']
        self.transform = transform
        if split == 'test':
            split = 'test_b'
        self.corrupted_paths = extract_images(os.path.join(root, split, 'data'))
        self.img_paths = extract_images(os.path.join(root, split, 'gt'))
        assert len(self.img_paths) == len(self.corrupted_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.corrupted_paths[item]).convert('RGB')
        gt_img = Image.open(self.img_paths[item]).convert('RGB')
        if self.transform is not None:
            X = self.transform(X)
            gt_img = self.transform(gt_img)
        return X, gt_img, None, None


def get_default_transforms(img_size: int, split: str):
    crop = T.RandomCrop if split == 'train' else T.CenterCrop
    transforms = T.Compose([
        T.Resize(img_size),
        crop((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return transforms


def _test():
    dataset = Raindrop(root='/data/Raindrop/', split='train')
    print(len(dataset))
    dataset = Raindrop(root='/data/Raindrop/', split='test')
    print(len(dataset))


if __name__ == '__main__':
    _test()
