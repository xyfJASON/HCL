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


class CelebA_HQ(Dataset):
    """
    The downloaded 30,000 images should be stored under `root/CelebA-HQ-img/`.

    The file names should be the same as their counterparts in the original CelebA dataset.

    The train/valid/test sets are split according to the original CelebA dataset,
    resulting in 24,183 training images, 2,993 validation images, and 2,824 test images.

    """
    def __init__(self, root, split='train', transform=None):
        assert split in ['train', 'valid', 'test', 'all']
        image_root = os.path.join(os.path.expanduser(root), 'CelebA-HQ-img')
        assert os.path.isdir(image_root), f'{image_root} is not an existing directory'

        self.transform = transform
        self.img_paths = extract_images(image_root)
        celeba_splits = [1, 162771, 182638, 202600]

        def filter_func(p):
            if split == 'all':
                return True
            k = 0 if split == 'train' else (1 if split == 'valid' else 2)
            return celeba_splits[k] <= int(os.path.splitext(os.path.basename(p))[0]) < celeba_splits[k+1]

        self.img_paths = list(filter(filter_func, self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.img_paths[item])
        if self.transform is not None:
            X = self.transform(X)
        return X


def get_default_transforms(img_size: int):
    transforms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return transforms


if __name__ == '__main__':
    dataset = CelebA_HQ(root='/data/CelebA-HQ/', split='train')
    print(len(dataset))
    dataset = CelebA_HQ(root='/data/CelebA-HQ/', split='valid')
    print(len(dataset))
    dataset = CelebA_HQ(root='/data/CelebA-HQ/', split='test')
    print(len(dataset))
    dataset = CelebA_HQ(root='/data/CelebA-HQ/', split='all')
    print(len(dataset))
