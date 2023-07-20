import os
from PIL import Image
from torch.utils.data import Dataset


class ImageDir(Dataset):
    def __init__(self, root, split='train', transform=None):
        assert split in ['train', 'val', 'valid', 'test']
        root = os.path.join(os.path.expanduser(root), split)
        assert os.path.isdir(root), f'{root} is not a valid directory'

        self.transform = transform

        img_ext = ['.png', '.jpg', '.jpeg']
        self.img_paths = []
        for curdir, subdirs, files in os.walk(root):
            for file in files:
                if os.path.splitext(file)[1].lower() in img_ext:
                    self.img_paths.append(os.path.join(curdir, file))
        self.img_paths = sorted(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.img_paths[item]).convert('RGB')
        if self.transform is not None:
            X = self.transform(X)
        return X


if __name__ == '__main__':
    dataset = ImageDir(root='/Users/jason/data/CelebA-HQ/')
    print(len(dataset))
