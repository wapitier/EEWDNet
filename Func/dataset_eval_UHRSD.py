import os
import glob
import skimage.io as io
import torch.utils.data as data
from torchvision.transforms import functional as F


class UHRSD_Dataset(data.Dataset):
    def __init__(self):

        val_set = '/data2021/tb/UHRSD'
        
        val_pth = glob.glob(os.path.join(val_set, 'val', 'Images', '*.jpg'))    
        self.images_path = val_pth


    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        file_name = image_path.split('/')[-1].split('.')[0]

        image = io.imread(image_path)
        h, w, _ = image.shape

        image = F.to_tensor(image)
        image = F.resize(image, (1024, 1024))
        image = F.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        return image, file_name, (h, w)

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)

        return batched_imgs


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)

    return batched_imgs

