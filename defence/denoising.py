import torch
import torch.nn as nn
import torch.optim as optim
from autoencoder import Autoencoder
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models
from torchvision import transforms
import cv2
import pywt
import numpy as np

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
                , torchvision.transforms.Resize((32,32),antialias=True)
                , torchvision.transforms.RandomCrop(32, padding=4)  # 先四周填充0，在吧图像随机裁剪成32*32
                , torchvision.transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转 选择一个概率概率
                , torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
            ]),
            'valid': torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
                , torchvision.transforms.Resize((32,32),antialias=True)
                , torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

gtsrb_train = datasets.GTSRB(root='/home/hx/code/data', split='train', transform=transforms['train'], download=True)
gtsrb_test = datasets.GTSRB(root='/home/hx/code/data', split='test', transform=transforms['valid'], download=True)


svhn_test = datasets.SVHN(root='/home/hx/code/data', split='test', transform=transforms['valid'], download=True)


train_loader = DataLoader(gtsrb_train, batch_size=64, shuffle=True,num_workers=8)
gtsrb_test_loader = DataLoader(gtsrb_test, batch_size=64,num_workers=8)
svhn_test_loader = DataLoader(svhn_test, batch_size=64,num_workers=8)


model = torchvision.models.resnet34(pretrained=False, num_classes=43) 
model.fc = nn.Sequential(nn.Dropout(0.4),nn.Linear(model.fc.in_features, model.fc.out_features))
model.load_state_dict(torch.load('/home/hx/code/CAMH/ckpt/best_model_GTSRB+SVHN.pth'))
model = model.to(device)

extra_layer = torch.nn.Linear(43, 10, bias=True)
extra_layer.load_state_dict(torch.load('/home/hx/code/CAMH/ckpt/best_layer_GTSRB+SVHN.pth'))
extra_layer = extra_layer.to(device)

noise = torch.load('/home/hx/code/CAMH/ckpt/best_noise_GTSRB+SVHN.pth')
noise = noise.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

autoencoder = Autoencoder().to(device)
autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
mse_loss = nn.MSELoss()


median_filter = torchvision.transforms.Compose([
    torchvision.transforms.GaussianBlur(kernel_size=3)
])


def bilateral_filter(image):

    img = image.cpu().numpy().transpose(1, 2, 0)

    filtered = cv2.bilateralFilter(img, 9, 75, 75)

    return torch.from_numpy(filtered.transpose(2, 0, 1)).to(device)


def wavelet_denoising(image):

    image_cpu = image.cpu().numpy()

    denoised_channels = []
    for channel in range(image_cpu.shape[0]):

        coeffs = pywt.wavedec2(image_cpu[channel], 'db1', level=2)

        cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
        
        threshold = 0.1
        coeffs_thresholded = [cA2,
                             (pywt.threshold(cH2, threshold, mode='soft'),
                              pywt.threshold(cV2, threshold, mode='soft'),
                              pywt.threshold(cD2, threshold, mode='soft')),
                             (pywt.threshold(cH1, threshold, mode='soft'),
                              pywt.threshold(cV1, threshold, mode='soft'),
                              pywt.threshold(cD1, threshold, mode='soft'))]
        
        denoised_channel = pywt.waverec2(coeffs_thresholded, 'db1')
        denoised_channels.append(denoised_channel)
    
    denoised_image = torch.tensor(np.stack(denoised_channels)).to(device).float()
    return denoised_image

def train_autoencoder(epochs=50):
    for epoch in range(epochs):
        autoencoder.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            
            noisy_images = images + torch.randn_like(images) * 0.1
            
            denoised = autoencoder(noisy_images)
            
            loss = mse_loss(denoised, images)
            
            autoencoder_optimizer.zero_grad()
            loss.backward()
            autoencoder_optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}')
    
    torch.save(autoencoder.state_dict(), 'denoising_autoencoder.pth')

def test(model, test_loader, dataset_type='gtsrb', denoising_method='none'):
    model.eval()
    autoencoder.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            if dataset_type == 'svhn':
                images = images + noise
                if denoising_method == 'median':
                    denoised_images = torch.stack([median_filter(img) for img in images])
                elif denoising_method == 'bilateral':
                    denoised_images = torch.stack([bilateral_filter(img) for img in images])
                elif denoising_method == 'wavelet':
                    denoised_images = torch.stack([wavelet_denoising(img) for img in images])
                elif denoising_method == 'autoencoder':
                    denoised_images = autoencoder(images)
                else:
                    denoised_images = images
                features = model(denoised_images)
                outputs = extra_layer(features)
            else:

                if denoising_method == 'autoencoder':
                    denoised_images = autoencoder(images)
                elif denoising_method == 'median':
                    denoised_images = torch.stack([median_filter(img) for img in images])
                elif denoising_method == 'bilateral':
                    denoised_images = torch.stack([bilateral_filter(img) for img in images])
                elif denoising_method == 'wavelet':
                    denoised_images = torch.stack([wavelet_denoising(img) for img in images])
                else:
                    denoised_images = images
                outputs = model(denoised_images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ == "__main__":
    denoise_methods = ['autoencoder', 'wavelet', 'median', 'bilateral']
    for method in denoise_methods:
        if method == 'autoencoder':
            train_autoencoder(epochs=50)
        print(f"\nTesting with {method} denoising...")
        gtsrb_acc = test(model, gtsrb_test_loader, 'gtsrb', method)
        svhn_acc = test(model, svhn_test_loader, 'svhn', method)
        print(f'GTSRB Accuracy: {gtsrb_acc:.2f}%')
        print(f'SVHN Accuracy: {svhn_acc:.2f}%')
        