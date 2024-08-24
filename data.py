import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import numpy as np

SIZE = 256
class ColorizationDataset(Dataset):
    def __init__(self, paths, mode = 'train'):
        self.paths = paths
        self.img_size = SIZE
        self.is_train = mode
        if mode == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), Image.BICUBIC),
                transforms.RandomHorizontalFlip()
            ])
        elif mode == 'val':
            self.transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), Image.BICUBIC)
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idxs):
        images = Image.OPEN(self.paths[idxs])

        img_rgb = images.convert('RGB')
        img_rgb = self.transforms(img_rgb)
        img_rgb = np.array(img_rgb)

        img_lab = rgb2lab(img_rgb).astype('float32')
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1
        ab = img_lab[[1 , 2], ...] / 110.

        return {'L': L, 'ab': ab}

def ColorizationDataLoader( batch_size = 32, n_workers = 4, pin_memory = True, **kwargs):
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle = True, num_workers=n_workers, pin_memory= pin_memory)
    return dataloader





