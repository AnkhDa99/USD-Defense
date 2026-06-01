# adaptive_attack_main.py (Corrected Version)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

# --- CRITICAL CHANGE: Import the custom ResNet model from the project's 'models' folder ---
from networks import*

# Import the SAM optimizer
from sam import SAM

# --- 1. 参数设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
poison_rate = 0.1
target_label = 0
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4
epochs = 50
rho = 0.02

# --- (The rest of the script remains the same) ---

# --- 2. 数据准备和毒化 ---
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trigger = torch.ones(3, 5, 5)


def add_trigger(image):
    image_copy = image.clone()
    image_copy[:, -6:-1, -6:-1] = trigger
    return image_copy


poison_indices = torch.randperm(len(trainset))[:int(len(trainset) * poison_rate)]
poisoned_data = []
for i, (image, label) in enumerate(trainset):
    if i in poison_indices:
        poisoned_data.append((add_trigger(image), target_label))
    else:
        poisoned_data.append((image, label))
trainloader = torch.utils.data.DataLoader(poisoned_data, batch_size=128, shuffle=True, num_workers=2)

# --- 3. 模型和优化器 ---
print("==> Building model...")
# --- CRITICAL CHANGE: Instantiate the custom ResNet-18 for CIFAR-10 ---
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)

base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=learning_rate, momentum=momentum,
                weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()


# --- 4. 训练循环 (Function) ---
def train(epoch, model, trainloader, criterion, optimizer):
    print(f'\nEpoch: {epoch}')
    model.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        criterion(model(inputs), targets).backward()
        optimizer.second_step(zero_grad=True)

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(trainloader)} | Loss: {loss.item():.3f}')


# --- 5. 主执行逻辑 ---
if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch, model, trainloader, criterion, optimizer)

    print("==> Saving smooth backdoor model...")
    save_dir = r'D:\pythontest\FIP\src\saved_models'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'smooth_backdoor_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")