import os
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T


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


class Places365(Dataset):
    def __init__(self, root, split='train', small=False, is_challenge=False, transform=None):
        size = '256' if small else 'large'
        variant = 'challenge' if is_challenge else 'standard'
        if split == 'train':
            img_root = os.path.join(root, f'data_{size}_{variant}')
        elif split == 'valid':
            img_root = os.path.join(root, f'val_{size}')
        elif split == 'test':
            img_root = os.path.join(root, f'test_{size}')
        else:
            raise ValueError
        assert os.path.isdir(img_root), f'{img_root} is not an existing directory'

        self.transform = transform
        self.img_paths = extract_images(img_root)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.img_paths[item]).convert('RGB')
        if self.transform is not None:
            X = self.transform(X)
        return X


def get_default_transforms(img_size: int, split: str):
    crop = T.RandomCrop if split == 'train' else T.CenterCrop
    transforms = T.Compose([
        T.Resize(img_size),
        crop((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return transforms


if __name__ == '__main__':
    dataset = Places365(root='/data/Places365/', split='train')
    print(len(dataset))
    dataset = Places365(root='/data/Places365/', split='valid')
    print(len(dataset))
    dataset = Places365(root='/data/Places365/', split='test')
    print(len(dataset))
