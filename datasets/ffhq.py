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


class FFHQ(Dataset):
    """
    The downloaded 70,000 images should be organized in the following structure:

    - root/
        - image1024x1024/
            - 00000/
                - 00000.png
                - 00001.png
                - ...
                - 00999.png
            - ...
            - 69000/
                - 69000.png
                - 69001.png
                - ...
                - 69999.png

    """
    def __init__(self, root, split='train', original_size=1024, transform=None):
        assert split in ['train', 'test', 'all']
        image_root = os.path.join(os.path.expanduser(root), f'images{original_size}x{original_size}')
        assert os.path.isdir(image_root), f'{image_root} is not an existing directory'

        self.transform = transform
        self.img_paths = extract_images(image_root)
        if split == 'train':
            self.img_paths = list(filter(lambda p: '00000' <= (os.path.dirname(p)).split('/')[-1] < '60000', self.img_paths))
        elif split == 'test':
            self.img_paths = list(filter(lambda p: '60000' <= (os.path.dirname(p)).split('/')[-1] < '70000', self.img_paths))

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
    dataset = FFHQ(root='/data/FFHQ/', split='train')
    print(len(dataset))
    dataset = FFHQ(root='/data/FFHQ/', split='test')
    print(len(dataset))
    dataset = FFHQ(root='/data/FFHQ/', split='all')
    print(len(dataset))
