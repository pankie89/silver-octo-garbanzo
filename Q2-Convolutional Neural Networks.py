import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 100
LEARNING_RATE = 0.001
MOMENTUM = 0.9
TEST_FREQUENCY = 100  # Number of training batches to go over before we test the model

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("data/", train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    ), batch_size=BATCH_SIZE, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("data/", train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    ), batch_size=BATCH_SIZE, shuffle=True
)

# 圖片可視化
def show_images():
    dataiter = iter(test_loader)
    imgs, labels = next(dataiter)
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.title(f"label: {labels[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_images()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # LeNet-5 論文、CS231n課程
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)  # Pool2：kernel 2x2，stride=2 ⇒ 16x4x4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

def test(net, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 訓練
for epoch in range(1):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % TEST_FREQUENCY == 0:
            acc = test(net, test_loader)
            print(f"Epoch {epoch+1}, Batch {i}, Accuracy: {acc:.2f}%")

acc = test(net, test_loader)
print("Final Accuracy: {:.2f}%".format(acc))
torch.save(net, 'model.pth')

# 預測6張圖片
net = torch.load('model.pth')

def predict_some():
    dataiter = iter(test_loader)
    imgs, labels = next(dataiter)
    with torch.no_grad():
        outputs = net(imgs[:6])
        _, predicted = torch.max(outputs, 1)

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.title(f"P:{predicted[i].item()} / T:{labels[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

predict_some()
