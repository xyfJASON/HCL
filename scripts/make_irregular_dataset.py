import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T


if __name__ == '__main__':
    downloaded_test = './downloaded_test/'
    save_path_train = './train/'
    save_path_test = './test/'

    masks = os.listdir(downloaded_test)
    for k in range(6):
        if not os.path.exists(os.path.join(save_path_train, str(k))):
            os.mkdir(os.path.join(save_path_train, str(k)))
        if not os.path.exists(os.path.join(save_path_test, str(k))):
            os.mkdir(os.path.join(save_path_test, str(k)))

        tests = np.random.choice(np.arange(2000), size=400, replace=False)
        for idx, fileid in enumerate(tqdm(range(k * 2000, (k + 1) * 2000))):
            mask = Image.open(os.path.join(downloaded_test, str(fileid).zfill(5)+'.png'))
            mask = T.ToTensor()(mask)
            # binarization
            mask = torch.where(mask < 0.5, 0., 1.)
            # ratation & flipping
            for i in range(4):
                for j in range(2):
                    tmp = T.RandomRotation((i * 90, i * 90))(mask)
                    tmp = T.RandomHorizontalFlip(p=j)(tmp)
                    tmp = T.ToPILImage()(tmp)

                    if idx in tests:
                        tmp.save(os.path.join(save_path_test, str(k), str(fileid).zfill(5)+f'_{i*2+j}.png'))
                    else:
                        tmp.save(os.path.join(save_path_train, str(k), str(fileid).zfill(5)+f'_{i*2+j}.png'))
