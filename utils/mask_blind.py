from PIL import ImageFilter
from typing import Tuple, List, Union

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from utils.mask import DatasetWithMask


class DatasetWithMaskBlind(DatasetWithMask):
    def __init__(
            self,
            dataset: Dataset,

            mask_type: Union[str, List[str]] = (),
            dir_path: str = None,
            dir_invert_color: bool = False,
            rect_num: Tuple[int, int] = (0, 4),
            rect_length_ratio: Tuple[float, float] = (0.2, 0.8),
            brush_num: Tuple[int, int] = (1, 9),
            brush_turns: Tuple[int, int] = (4, 18),
            brush_width_ratio: Tuple[float, float] = (0.02, 0.1),
            brush_length_ratio: Tuple[float, float] = (0.1, 0.25),

            noise_type: str = 'constant',
            constant_value: Tuple[float, float, float] = (0, 0, 0),
            real_dataset: Dataset = None,
            smooth_radius: float = 0.0,

            is_train: bool = False,
    ):
        super().__init__(
            dataset=dataset,
            mask_type=mask_type,
            dir_path=dir_path,
            dir_invert_color=dir_invert_color,
            rect_num=rect_num,
            rect_length_ratio=rect_length_ratio,
            brush_num=brush_num,
            brush_turns=brush_turns,
            brush_width_ratio=brush_width_ratio,
            brush_length_ratio=brush_length_ratio,
            is_train=is_train,
        )
        self.noise_type = noise_type
        self.constant_value = constant_value
        self.real_dataset = real_dataset
        self.smooth_radius = smooth_radius
        self.is_train = is_train

    def __getitem__(self, item):
        image = self.dataset[item]
        image = image[0] if isinstance(image, (tuple, list)) else image
        C, H, W = image.shape
        mask = self.mask_generator.sample(int(H), int(W), item)
        noise = self.sample_noise(H, W, item)
        smooth_mask = mask.float()
        if self.smooth_radius > 0:
            smooth_mask = T.ToPILImage()(smooth_mask)
            smooth_mask = smooth_mask.filter(ImageFilter.GaussianBlur(radius=self.smooth_radius))
            smooth_mask = T.ToTensor()(smooth_mask)
        corrupted_image = smooth_mask * image + (1 - smooth_mask) * noise
        return corrupted_image, image, noise, mask

    def sample_noise(self, H: int, W: int, item: int):
        if isinstance(item, torch.Tensor):
            item = item.item()
        if self.is_train is False:
            rndgn = torch.Generator()
            rndgn.manual_seed(item + 3407)
        else:
            rndgn = torch.default_generator

        if self.noise_type == 'constant':
            noise = torch.tensor(self.constant_value).view(-1, 1, 1).expand(3, H, W)
        elif self.noise_type == 'random_noise':
            noise = torch.rand(3, H, W) * 2 - 1
        elif self.noise_type == 'random_single_color':
            noise = torch.rand((3, 1, 1), generator=rndgn).expand(3, H, W) * 2 - 1
        elif self.noise_type == 'real':
            idx = torch.randint(0, len(self.real_dataset), (1, ), generator=rndgn).item()  # type: ignore
            noise = self.real_dataset[idx]
            noise = noise[0] if isinstance(noise, (tuple, list)) else noise
            noise = T.RandomCrop((H, W))(noise)
        else:
            raise ValueError
        return noise
