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


class ParisStreetView(Dataset):
    """
    The downloaded data should be organized in the following structure:

    - root/
        - paris_train_original/ (14,900 images extracted from paris_train_original.zip)
            - 48.842502_2.344968_90_-004.JPG
            - ...
        - paris_eval_gt/ (100 images extracted from paris_eval_75876.zip)
            - 001_im.png
            - ...

    """
    def __init__(self, root, split='train', transform=None):
        assert split in ['train', 'test', 'all']
        train_root = os.path.join(root, 'paris_train_original')
        eval_root = os.path.join(root, 'paris_eval_gt')
        assert os.path.isdir(root) and os.path.isdir(train_root) and os.path.isdir(eval_root)

        self.transform = transform
        self.img_paths = []
        if split in ['train', 'all']:
            self.img_paths.extend(extract_images(train_root))
        if split in ['test', 'all']:
            self.img_paths.extend(extract_images(eval_root))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.img_paths[item])
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
    dataset = ParisStreetView(root='/data/ParisStreetView', split='train')
    print(len(dataset))
    dataset = ParisStreetView(root='/data/ParisStreetView', split='test')
    print(len(dataset))
    dataset = ParisStreetView(root='/data/ParisStreetView', split='all')
    print(len(dataset))
