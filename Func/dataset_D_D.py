import os
import glob
import skimage.io as io
import torch.utils.data as data


class SOD_Dataset(data.Dataset):
    def __init__(self, train: bool = True, transform=None):

        train_set_1 = '/data2021/tb/DUTS_1024'

        val_set = '/data2021/tb/DUTS_1024'

        if train:
            tr1_path = glob.glob(os.path.join(train_set_1, 'train', 'Images', '*.jpg'))
            all_img_path = tr1_path

            all_msk_path = []
            for pth in all_img_path:
                sub = pth.split('Images')
                file_name = sub[1].split('.')[0] + '.png'
                all_msk_path.append(sub[0] + 'Masks' + file_name)

        else:
            val_pth = glob.glob(os.path.join(val_set, 'val', 'Images', '*.jpg'))    
            all_img_path = val_pth

            all_msk_path = []
            for pth in all_img_path:
                sub = pth.split('Images')
                file_name = sub[1].split('.')[0] + '.png'
                all_msk_path.append(sub[0] + 'Masks' + file_name)

        self.images_path = all_img_path
        self.masks_path = all_msk_path
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        image = io.imread(image_path)
        assert image is not None, f"failed to read image: {image_path}"
        h, w, _ = image.shape

        target = io.imread(mask_path)
        assert target is not None, f"failed to read mask: {mask_path}"
        if (len(target.shape) > 2):
            target = target[:, :, 0]

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


