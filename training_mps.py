import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler, optimizer
import torchvision
import os, sys
import cv2 as cv
from torch.utils.data import DataLoader, sampler

class SegmentationDataset(object):
    def __init__(self, image_dir, mask_dir, image_size=(320, 480)):
        self.images = []
        self.masks = []
        self.image_size = image_size
        files = os.listdir(image_dir)
        sfiles = os.listdir(mask_dir)
        for i in range(len(sfiles)):
            img_file = os.path.join(image_dir, files[i])
            mask_file = os.path.join(mask_dir, sfiles[i])
            # print(img_file, mask_file)
            self.images.append(img_file)
            self.masks.append(mask_file)

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if isinstance(idx, list):
            images = []
            masks = []
            for i in idx:
                img, mask = self.load_sample(i)
                images.append(img)
                masks.append(mask)
            batch = {'image': torch.stack(images), 'mask': torch.stack(masks)}
            return batch
        else:
            img, mask = self.load_sample(idx)
            sample = {'image': img, 'mask': mask}
            return sample

    def load_sample(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  # BGR order
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        img = cv.resize(img, self.image_size)
        mask = cv.resize(mask, self.image_size)

        # Normalize the image
        img = np.float32(img) / 255.0
        img = np.expand_dims(img, 0)

        # Process the mask
        mask[mask <= 128] = 0
        mask[mask > 128] = 1
        mask = np.expand_dims(mask, 0)

        return torch.from_numpy(img), torch.from_numpy(mask)

class UNetModel(torch.nn.Module):

    def __init__(self, in_features=1, out_features=2, init_features=32):
        super(UNetModel, self).__init__()
        features = init_features
        self.encode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_features, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU()
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features*2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU()
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU()
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_decode_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*16, out_channels=features*16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 16),
            torch.nn.ReLU()
        )
        self.upconv4 = torch.nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decode_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*16, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features*8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decode_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU()
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU()
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU()
        )
        self.out_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=out_features, kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        enc1 = self.encode_layer1(x)
        enc2 = self.encode_layer2(self.pool1(enc1))
        enc3 = self.encode_layer3(self.pool2(enc2))
        enc4 = self.encode_layer4(self.pool3(enc3))

        bottleneck = self.encode_decode_layer(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decode_layer4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decode_layer3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decode_layer2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decode_layer1(dec1)

        out = self.out_layer(dec1)
        return out
    
def collate_fn(batch):
    images = [item['image'] for item in batch]
    masks = [item['mask'] for item in batch]
    images = torch.stack(images)
    masks = torch.stack(masks)
    return {'image': images, 'mask': masks}

def train():
    if __name__ == '__main__':
        index = 0
        num_epochs = 10
        train_on_gpu = True
        unet = UNetModel()
        # model_dict = unet.load_state_dict(torch.load('unet_road_model-100.pt'))

        image_dir = 'CrackForest-dataset-master/image/'
        mask_dir = 'CrackForest-dataset-master/groundTruthPngImg/'

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        unet.to(device)

        dataloader = SegmentationDataset(image_dir, mask_dir)

        optimizer = torch.optim.SGD(unet.parameters(), lr=0.01, momentum=0.9)

        train_loader = DataLoader(
            dataloader, batch_size=4, shuffle=True, collate_fn=collate_fn)
        
        for batch in train_loader:
            images, masks = batch['image'], batch['mask']
        
        for epoch in range(num_epochs):
            train_loss = 0.0
            for batch in train_loader:
                images_batch, target_labels = \
                    images, masks

                print(target_labels.min())
                print(target_labels.max())

                if train_on_gpu:
                    images_batch, target_labels = images_batch.to(device), target_labels.to(device)
                optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                m_label_out_ = unet(images_batch)
                # print(m_label_out_)
                # calculate the batch loss
                target_labels = target_labels.contiguous().view(-1)
                m_label_out_ = m_label_out_.view(-1,2)
                target_labels = target_labels.long()

                m_label_out_.to(device)
                target_labels.to(device)
                criterion = torch.nn.BCEWithLogitsLoss()
                loss = torch.nn.functional.cross_entropy(m_label_out_, target_labels)
                loss.to(device)
                print(loss)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                optimizer.step()

                # update training loss
                train_loss += loss.item()
                if index % 100 == 0:
                    print('step: {} \tcurrent Loss: {:.6f} '.format(index, loss.item()))
                index += 1
                # test(unet)
            # 计算平均损失
            train_loss = train_loss / len(train_loader)
            # 显示训练集与验证集的损失函数
            print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))
            # test(unet)
        # save model
        unet.eval()
        torch.save(unet.state_dict(), 'unet_road_model.pt')
    
train()

def test(unet):
    model_dict=unet.load_state_dict(torch.load('unet_road_model.pt'))
    root_dir = 'CrackForest-dataset-master/test/'
    fileNames = os.listdir(root_dir)
    for f in fileNames:
        image = cv.imread(os.path.join(root_dir, f), cv.IMREAD_GRAYSCALE)
        h, w = image.shape
        img = np.float32(image) /255.0
        img = np.expand_dims(img, 0)
        x_input = torch.from_numpy(img).view( 1, 1, h, w)
        probs = unet(x_input.cuda())
        m_label_out_ = probs.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
        grad, output = m_label_out_.data.max(dim=1)
        output[output > 0] = 255
        predic_ = output.view(h, w).cpu().detach().numpy()

        # print(predic_)
        # print(predic_.max())
        # print(predic_.min())

        # print(predic_)
        # print(predic_.shape)
        # cv.imshow("input", image)
        result = cv.resize(np.uint8(predic_), (w, h))

        cv.imshow("unet-segmentation-demo", result)
        cv.waitKey(0)
    cv.destroyAllWindows()

def testPt(unet):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_dict=unet.load_state_dict(torch.load('unet_road_model.pt'))
    root_dir = 'CrackForest-dataset-master/test/'
    fileNames = os.listdir(root_dir)
    for f in fileNames:
        image = cv.imread(os.path.join(root_dir, f), cv.IMREAD_GRAYSCALE)
        h, w = image.shape
        img = np.float32(image) /255.0
        img = np.expand_dims(img, 0)
        x_input = torch.from_numpy(img).view( 1, 1, h, w)
        x_input.to(device)
        probs = unet(x_input)
        probs.to(device)
        m_label_out_ = probs.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
        grad, output = m_label_out_.data.max(dim=1)
        output[output > 0] = 255
        predic_ = output.view(h, w).cpu().detach().numpy()

        # print(predic_)
        # print(predic_.max())
        # print(predic_.min())

        # print(predic_)
        # print(predic_.shape)
        # cv.imshow("input", image)
        result = cv.resize(np.uint8(predic_), (w, h))

        cv.imshow("unet-segmentation-demo", result)
        cv.waitKey(0)
    cv.destroyAllWindows()

unet = UNetModel()
testPt(unet)
