import torchvision.transforms as T
import torchvision.datasets as dset


CelebA = dset.CelebA


def get_default_transforms(img_size: int):
    transforms = T.Compose([
        T.CenterCrop((140, 140)),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return transforms
