import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from numpy.lib.stride_tricks import as_strided
import torch
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

SIZE = 256

preprocess = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor()
])


def loader(path):
    img_pil = Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor


def pool2d(A, kernel_size, stride):
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride * A.strides[0],
                              stride * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)
    return A_w.mean(axis=(1, 2)).reshape(output_shape)


def Calsimup(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m - 1, n))
    for i in range(m - 1):
        for j in range(n):
            if abs(x[i + 1, j] - x[i, j]) > 0:
                ret[i, j] = 0
    return ret


def Calsimleft(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m, n - 1))
    for i in range(m):
        for j in range(n - 1):
            if abs(x[i, j + 1] - x[i, j]) > 0:
                ret[i, j] = 0
    return ret


def Calsimup_bank(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m, n + 2))
    for i in range(2, m):
        for j in range(1, n - 1):
            if abs(x[i, j] - x[i - 2, j]) > 0:
                ret[i, j] = 0
    return ret


def Calsimleft_bank(x):
    m, n = x.shape[0], x.shape[1]
    ret = np.ones((m + 2, n))
    for i in range(1, m - 1):
        for j in range(2, n):
            if abs(x[i, j] - x[i, j - 2]) > 0:
                ret[i, j] = 0
    return ret


class DealDataset(Dataset):
    def __init__(self, TRAIN_FAKE_ROOT, TRAIN_REAL_ROOT, LENGTH, TYPE, loader=loader,train=True):
        self.len = LENGTH
        self.loader = loader
        self.fake_root = TRAIN_FAKE_ROOT
        self.real_root = TRAIN_REAL_ROOT
        self.TYPE = TYPE
        self.train = train
        train_fake_video_paths = os.listdir(self.fake_root)

        self.train_fake_imgs = []
        for i in train_fake_video_paths:
            video_path = self.fake_root + i
            img = os.listdir(video_path)
            self.train_fake_imgs.append([video_path + '/' + j for j in img])

        train_real_video_paths = os.listdir(self.real_root)
        self.train_real_imgs = []
        for i in train_real_video_paths:
            video_path = self.real_root + i
            img = os.listdir(video_path)
            self.train_real_imgs.append([video_path + '/' + j for j in img])
        self.NUM_fake = len(self.train_fake_imgs)
        self.NUM_real = len(self.train_real_imgs)

    def __getitem__(self, index):
        if np.random.randint(0, 2):
            video_index = np.random.randint(0, self.NUM_fake)
            img_index = np.random.randint(0, len(self.train_fake_imgs[video_index]))
            img_path = self.train_fake_imgs[video_index][img_index]
            img = self.loader(img_path)

            if self.train:
                mask_path = img_path.replace(self.TYPE, 'mask')
                fake_mask = cv2.imread(mask_path, 0)
                
                fake_mask = np.array(cv2.resize(fake_mask, (SIZE, SIZE)) > 1, dtype=np.float64)
                fake_mask1 = pool2d(fake_mask, 16, 16)

                fake_mask_up = Calsimup(fake_mask1)
                fake_mask_left = Calsimleft(fake_mask1)
                fake_mask_up = torch.from_numpy(np.expand_dims(fake_mask_up, 0))
                fake_mask_left = torch.from_numpy(np.expand_dims(fake_mask_left, 0))
                mask_up = torch.tensor(fake_mask_up, dtype=torch.float32)
                mask_left = torch.tensor(fake_mask_left, dtype=torch.float32)

                fake_mask_up_bank = Calsimup_bank(fake_mask1)
                fake_mask_left_bank = Calsimleft_bank(fake_mask1)
                fake_mask_up_bank = torch.from_numpy(np.expand_dims(fake_mask_up_bank, 0))
                fake_mask_left_bank = torch.from_numpy(np.expand_dims(fake_mask_left_bank, 0))
                mask_up_bank = torch.tensor(fake_mask_up_bank, dtype=torch.float32)
                mask_left_bank = torch.tensor(fake_mask_left_bank, dtype=torch.float32)
            else:
                mask_up, mask_up_bank, mask_left, mask_left_bank = 0,0,0,0
            label = torch.tensor([1])

        else:
            video_index = np.random.randint(0, self.NUM_real)
            img_index = np.random.randint(0, len(self.train_real_imgs[video_index]))
            img_path = self.train_real_imgs[video_index][img_index]
            img = self.loader(img_path)
            if self.train:
                mask_up = torch.ones((1, 15, 16), dtype=torch.float32)
                mask_left = torch.ones((1, 16, 15), dtype=torch.float32)
                mask_up_bank = torch.ones((1, 16, 18), dtype=torch.float32)
                mask_left_bank = torch.ones((1, 18, 16), dtype=torch.float32)
            else:
                mask_up, mask_up_bank, mask_left, mask_left_bank = 0,0,0,0
            label = torch.tensor([0])

        return img, (mask_up, mask_up_bank, mask_left, mask_left_bank,label)

    def __len__(self):
        return self.len


def Val(model, VAL_FAKE_ROOT, VAL_REAL_ROOT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ValDataset = DealDataset(VAL_FAKE_ROOT, VAL_REAL_ROOT,LENGTH =32*100, TYPE = ' ',train=False)
    val_loader = DataLoader(dataset=ValDataset, batch_size=32, shuffle=True)
    model.eval()
    acc_total = 0
    _labels = []
    rets = []
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            input, _, _, _, _, label = inputs.to(device), labels[0],labels[1],labels[2],labels[3],labels[4].to(device)
            output = model(input)
            acc = torch.sum(torch.eq(torch.ge(torch.sigmoid(output[4]), torch.full_like(output[4], 0.5)), label))
            acc_total += acc.cpu().numpy() / 32

            rets += list(output[4].detach().cpu().numpy())
            _labels += list(label.cpu().numpy())

    return acc_total/100




