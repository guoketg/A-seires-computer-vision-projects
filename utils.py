from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import GradScaler, autocast  # 关键修改：替换为 torch.amp
import time
import torch


# 2. 数据加载函数（无需修改，保持原样）
def load_data_fashion_mnist(batch_size, path, resize):
    Transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    trainSet = datasets.FashionMNIST(path, train=True, transform=Transform, download=True)
    testSet = datasets.FashionMNIST(path, train=False, transform=Transform, download=True)
    train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def load_data_mnist(batch_size, path, resize):
    Transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    trainSet = datasets.MNIST(path, train=True, transform=Transform, download=True)
    testSet = datasets.MNIST(path, train=False, transform=Transform, download=True)
    train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def load_data_cifar10(batch_size, path, resize):
    TrainTransform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(resize, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    TestTransform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    trainSet = datasets.CIFAR10(path, train=True, transform=TrainTransform, download=True)
    testSet = datasets.CIFAR10(path, train=False, transform=TestTransform, download=True)
    train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testSet, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# 3. GradScaler 初始化：指定 device="cuda"（关键修改）
scaler = GradScaler(device="cuda")


# 4. train 函数：autocast 增加 device_type="cuda"（关键修改）
def train(net, train_loader, test_loader, num_epochs, loss_fn1, optimizer1, device):
    net.to(device)
    print(f"training on {device}")
    optimizer = optimizer1
    loss_fn = loss_fn1
    total_time = 0
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        start = time.time()
        for batch_index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # 关键修改：autocast 增加 device_type 参数
            with autocast(device_type="cuda"):
                outputs = net(inputs)
                loss_value = loss_fn(outputs, labels)
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss_value.item()
            _, predicted = torch.max(outputs.detach(), 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
        end = time.time()
        print(
            f"training: epoch {epoch + 1} takes {end - start:.2f}s, accuracy {correct / total:.4f}, loss {train_loss / len(train_loader):.4f}")
        total_time += end - start

        # 测试阶段（无需修改）
        net.eval()
        start = time.time()
        test_total = 0
        test_accuracy = 0
        test_loss = 0
        with torch.no_grad():
            for batch_index, (test_image, test_tag) in enumerate(test_loader):
                test_image, test_tag = test_image.to(device), test_tag.to(device)
                test_total += test_image.shape[0]
                outputs = net(test_image)
                loss_value = loss_fn(outputs, test_tag)
                test_loss += loss_value.item()
                _, predicted = torch.max(outputs.detach(), 1)
                test_accuracy += (predicted == test_tag).sum().item()
        end = time.time()
        print(
            f"testing: epoch {epoch + 1} takes {end - start:.2f}s, accuracy {test_accuracy / test_total:.4f}, loss {test_loss / len(test_loader):.4f}")
        total_time += end - start
    print(f"whole process takes {total_time:.2f}s")