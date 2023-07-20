import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models


class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        if 'VGG19_Weights' in dir(models):
            vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        else:
            vgg19 = models.vgg19(pretrained=True)
        vgg19.eval()
        for param in vgg19.parameters():
            param.requires_grad_(False)

        self.relu1_2 = vgg19.features[:4]
        self.relu2_2 = vgg19.features[4:9]
        self.relu3_2 = vgg19.features[9:14]
        self.relu4_2 = vgg19.features[14:23]
        self.relu5_2 = vgg19.features[23:32]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, X: Tensor):
        """
        Args:
            X: input image ranging in [-1, 1]
        """
        X = (X + 1) / 2  # [0, 1]
        X = (X - self.mean) / self.std
        out = {}
        for i in range(1, 6):
            X = getattr(self, f'relu{i}_2')(X)
            out[f'relu{i}_2'] = X
        return out


def _test():
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    print(vgg)

    vgg_feature = VGG19FeatureExtractor()
    X = torch.randn(1, 3, 256, 256)
    out = vgg_feature(X)
    for k, v in out.items():
        print(k, v.shape)


if __name__ == '__main__':
    _test()
