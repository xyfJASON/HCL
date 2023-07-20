import os
import math
from PIL import Image, ImageDraw
from typing import Tuple, List, Union

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class DatasetWithMask(Dataset):
    def __init__(
            self,
            dataset,
            mask_type: Union[str, List[str]] = (),
            dir_path: str = None,
            dir_invert_color: bool = False,
            rect_num: Tuple[int, int] = (0, 4),
            rect_length_ratio: Tuple[float, float] = (0.2, 0.8),
            brush_num: Tuple[int, int] = (1, 9),
            brush_turns: Tuple[int, int] = (4, 18),
            brush_width_ratio: Tuple[float, float] = (0.02, 0.1),
            brush_length_ratio: Tuple[float, float] = (0.1, 0.25),
            is_train: bool = False,
    ):
        self.dataset = dataset
        self.mask_generator = MaskGenerator(
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image = self.dataset[item]
        image = image[0] if isinstance(image, (tuple, list)) else image
        C, H, W = image.shape
        mask = self.mask_generator.sample(int(H), int(W), item)
        return image, mask


class MaskGenerator:
    def __init__(
            self,
            mask_type: Union[str, List[str]] = (),
            dir_path: str = None,                                       # dir
            dir_invert_color: bool = False,                             # dir
            center_length_ratio: Tuple[float, float] = (0.25, 0.25),    # center
            rect_num: Tuple[int, int] = (0, 4),                         # rect
            rect_length_ratio: Tuple[float, float] = (0.2, 0.8),        # rect
            brush_num: Tuple[int, int] = (1, 9),                        # brush
            brush_turns: Tuple[int, int] = (4, 18),                     # brush
            brush_width_ratio: Tuple[float, float] = (0.02, 0.1),       # brush
            brush_length_ratio: Tuple[float, float] = (0.1, 0.25),      # brush
            is_train: bool = False,
    ):
        self.mask_type = mask_type
        self.dir_invert_color = dir_invert_color
        self.center_length_ratio = center_length_ratio
        self.rect_num = rect_num
        self.rect_length_ratio = rect_length_ratio
        self.brush_num = brush_num
        self.brush_turns = brush_turns
        self.brush_width_ratio = brush_width_ratio
        self.brush_length_ratio = brush_length_ratio
        self.is_train = is_train

        if isinstance(mask_type, str):
            self.mask_type = [mask_type]
        if 'dir' in self.mask_type:
            dir_path = os.path.expanduser(dir_path)
            assert os.path.isdir(dir_path), f'{dir_path} is not a valid directory'
            img_ext = ['.png', '.jpg', '.jpeg']
            self.mask_paths = []
            for curdir, subdir, files in os.walk(dir_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in img_ext:
                        self.mask_paths.append(os.path.join(curdir, file))
            self.mask_paths = sorted(self.mask_paths)

    def sample(self, H: int, W: int, item: int = None):
        if isinstance(item, torch.Tensor):
            item = item.item()
        if self.is_train is False:
            rndgn = torch.Generator()
            rndgn.manual_seed(item + 3407)
        else:
            rndgn = torch.default_generator

        mask = torch.ones((1, H, W), dtype=torch.bool)
        for t in self.mask_type:
            if t == 'dir':
                m = self._sample_dir(H, W, rndgn)
            elif t == 'center':
                m = self._sample_center(H, W, rndgn)
            elif t == 'rect':
                m = self._sample_rectangles(H, W, rndgn)
            elif t == 'brush':
                m = self._sample_brushes(H, W, rndgn)
            elif t == 'half':
                m = self._sample_half(H, W, rndgn)
            elif t == 'every-second-line':
                m = self._sample_every_second_line(H, W)
            elif t == 'sr2x':
                m = self._sample_sr2x(H, W)
            else:
                raise ValueError(f'mask type should be one of {{dir, center, rect, brush, '
                                 f'half, every-second-line, sr2x}}, get {t}')
            mask = torch.logical_and(mask, m)
        return mask

    def _sample_dir(self, H: int, W: int, rndgn: torch.Generator):
        path = self.mask_paths[torch.randint(0, len(self.mask_paths), (1, ), generator=rndgn).item()]
        mask = Image.open(path)
        mask = T.Resize((H, W))(mask)
        mask = T.ToTensor()(mask)
        if self.dir_invert_color:
            mask = torch.where(mask < 0.5, 1., 0.).bool()
        else:
            mask = torch.where(mask < 0.5, 0., 1.).bool()
        return mask

    def _sample_center(self, H: int, W: int, rndgn: torch.Generator):
        mask = torch.ones((1, H, W)).float()
        min_ratio, max_ratio = self.center_length_ratio
        ratio = torch.rand((1, ), generator=rndgn).item() * (max_ratio - min_ratio) + min_ratio
        h, w = int(ratio * H), int(ratio * W)
        mask[:, H//2-h//2:H//2+h//2, W//2-w//2:W//2+w//2] = 0.
        return mask.bool()

    def _sample_rectangles(self, H: int, W: int, rndgn: torch.Generator):
        min_num, max_num = self.rect_num
        min_ratio, max_ratio = self.rect_length_ratio
        n_rect = torch.randint(min_num, max_num + 1, (1, ), generator=rndgn).item()
        min_h, max_h = int(min_ratio * H), int(max_ratio * H)
        min_w, max_w = int(min_ratio * W), int(max_ratio * W)
        mask = torch.ones((1, H, W)).float()
        for i in range(n_rect):
            h = torch.randint(min_h, max_h + 1, (1, ), generator=rndgn).item()
            w = torch.randint(min_w, max_w + 1, (1, ), generator=rndgn).item()
            y = torch.randint(0, H - h + 1, (1, ), generator=rndgn).item()
            x = torch.randint(0, W - w + 1, (1, ), generator=rndgn).item()
            mask[:, y:y+h, x:x+w] = 0.
        return mask.bool()

    def _sample_brushes(self, H: int, W: int, rndgn: torch.Generator):
        min_num, max_num = self.brush_num
        min_turns, max_turns = self.brush_turns
        min_width = int(self.brush_width_ratio[0] * min(H, W))
        max_width = int(self.brush_width_ratio[1] * min(H, W))
        min_length = int(self.brush_length_ratio[0] * min(H, W))
        max_length = int(self.brush_length_ratio[1] * min(H, W))
        min_angle, max_angle = 4 * math.pi / 15, 8 * math.pi / 15
        n_brush = torch.randint(min_num, max_num + 1, (1, ), generator=rndgn).item()
        mask = Image.new('L', (W, H), 255)
        for i in range(n_brush):
            vertex = [(torch.randint(0, W, (1, ), generator=rndgn).item(),
                       torch.randint(0, H, (1, ), generator=rndgn).item())]
            n_turns = torch.randint(min_turns, max_turns + 1, (1, ), generator=rndgn).item()
            width = torch.randint(min_width, max_width + 1, (1, ), generator=rndgn).item()
            for j in range(n_turns + 1):
                angle = torch.rand(1, generator=rndgn).item() * (max_angle - min_angle) + min_angle
                if j % 2 == 0:
                    angle = 2 * math.pi - angle
                length = torch.randint(min_length, max_length + 1, (1, ), generator=rndgn).item()
                new_x = min(max(vertex[-1][0] + length * math.cos(angle), 0), W)
                new_y = min(max(vertex[-1][1] + length * math.sin(angle), 0), H)
                vertex.append((new_x, new_y))
            draw = ImageDraw.Draw(mask)
            draw.line(vertex, fill=0, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2), fill=0)
            if torch.rand(1, generator=rndgn) > 0.5:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # noqa
            if torch.rand(1, generator=rndgn) > 0.5:
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)  # noqa
        if torch.rand(1, generator=rndgn) > 0.5:
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)      # noqa
        if torch.rand(1, generator=rndgn) > 0.5:
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)      # noqa
        mask = T.ToTensor()(mask)
        mask = torch.where(mask < 0.5, 0., 1.).bool()
        return mask

    @staticmethod
    def _sample_half(H: int, W: int, rndgn: torch.Generator):
        mask = torch.ones((1, H, W)).float()
        direction = torch.randint(0, 4, (1, ), generator=rndgn).item()
        if direction == 0:
            mask[:, :H//2, :] = 0.
        elif direction == 1:
            mask[:, H//2:, :] = 0.
        elif direction == 2:
            mask[:, :, :W//2] = 0.
        else:
            mask[:, :, W//2:] = 0.
        return mask.bool()

    @staticmethod
    def _sample_every_second_line(H: int, W: int):
        mask = torch.ones((1, H, W)).float()
        mask[:, ::2, :] = 0.
        return mask.bool()

    @staticmethod
    def _sample_sr2x(H: int, W: int):
        mask = torch.ones((1, H, W)).float()
        mask[:, ::2, :] = 0.
        mask[:, :, ::2] = 0.
        return mask.bool()


def _test(**kwargs):
    from torchvision.utils import make_grid
    show = []
    mask_gen = MaskGenerator(**kwargs, is_train=True)
    for i in range(5):
        mask = mask_gen.sample(512, 512, item=0).float()
        show.append(mask)
    mask_gen = MaskGenerator(**kwargs, is_train=False)
    for i in range(5):
        mask = mask_gen.sample(512, 512, item=0).float()
        show.append(mask)
    show = make_grid(show, nrow=5)
    T.ToPILImage()(show).show()


if __name__ == '__main__':
    _test(mask_type='center', center_length_ratio=(0.25, 0.5))
    # _test(mask_type=['brush', 'rect'])
    # _test(mask_type=['half', 'sr2x'])
    # _test(mask_type=['every-second-line'])
    # _test(mask_type='dir',
    #       dir_path='/Volumes/Samsung PSSD T7 Media/data/NVIDIAIrregularMaskDataset/train/',
    #       dir_invert_color=True)
