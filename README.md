### **逐步解說：用 CNN 分類 MNIST 手寫數字**

#### **1. 導入必要工具包**
```python
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
```
- **Why needed ？**
  - `torch`: PyTorch 的核心，處理張量（多維陣列）和自動計算梯度。
  - `torchvision`: 提供常用數據集（如 MNIST）和圖像轉換工具。
  - `nn` 和 `functional`: 定義神經網絡層和激活函數（如 ReLU）。
  - `optim`: 包含優化算法（如 SGD），用於調整模型參數。
  - `matplotlib`: 畫圖顯示圖片和結果，方便直觀理解。

#### **2. 設定超參數**
```python
BATCH_SIZE = 100
LEARNING_RATE = 0.001
MOMENTUM = 0.9
TEST_FREQUENCY = 100
```
- **這些參數的意義：**
  - `BATCH_SIZE=100`: 每次訓練用 100 張圖片更新一次模型。批次越大，記憶體需求越高，但訓練更穩定。
  - `LEARNING_RATE=0.001`: 控制參數調整的步幅。太小會學得慢，太大可能無法收斂。
  - `MOMENTUM=0.9`: 加速梯度下降，避免卡在局部最小值。類似「慣性」效果。
  - `TEST_FREQUENCY=100`: 每訓練 100 個批次後測試一次模型，觀察進步情況。

#### **3. 加載 MNIST 數據集**
```python
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("data/", train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    ), batch_size=BATCH_SIZE, shuffle=True
)
```
- **關鍵步驟解析：**
  - `ToTensor()`: 將圖片從 PIL 格式轉為 PyTorch 張量（形狀為 [通道, 高, 寬]）。
  - `Normalize(均值, 標準差)`: 歸一化數據，加速訓練。MNIST 的均值和標準差是預先計算好的。
  - `shuffle=True`: 打亂數據順序，避免模型學習到順序相關的偏差。
- **為什麼要歸一化？**  
  將像素值從 [0,255] 縮放到 [-1,1] 附近，使訓練更穩定。

#### **4. 可視化測試集圖片**
```python
def show_images():
    dataiter = iter(test_loader)
    imgs, labels = next(dataiter)
    # 畫出 6 張圖
    ...
show_images()
```
- **代碼作用：**  
  顯示測試集中的 6 張圖片，確認數據加載正確。
- **為什麼用測試集？**  
  訓練前先觀察數據樣式，確保無異常（例如標籤錯誤）。

#### **5. 定義 LeNet-5 網絡結構**
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 第一次卷積+池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二次卷積+池化
        x = x.view(-1, 16*4*4)               # 展平為一維向量
        x = F.relu(self.fc1(x))               # 全連接層 1
        x = F.relu(self.fc2(x))               # 全連接層 2
        x = self.fc3(x)                      # 輸出層
        return x
```
- **各層設計原因：**
  - **卷積層 (Conv2d)**：提取局部特徵。第一層用 5x5 卷積核，從單通道（灰度）提取 6 種特徵。
  - **池化層 (MaxPool2d)**：降低維度，保留主要特徵。2x2 窗口，步長 2，將特徵圖尺寸減半。
  - **全連接層 (Linear)**：將特徵映射到 10 個類別（數字 0-9）。
- **維度變化示例：**
  - 輸入圖片：1x28x28 → 卷積1 → 6x24x24 → 池化 → 6x12x12  
  - 卷積2 → 16x8x8 → 池化 → 16x4x4 → 展平為 16*4*4=256 個神經元。

#### **6. 初始化模型、損失函數和優化器**
```python
net = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
```
- **損失函數選擇：**  
  `CrossEntropyLoss` 直接適用於多分類問題，內部已包含 Softmax。
- **優化器設定：**  
  SGD（隨機梯度下降）搭配動量（Momentum），加速收斂並減少震盪。

#### **7. 定義測試函數**
```python
def test(net, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():  # 不計算梯度，節省資源
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
```
- **代碼解析：**
  - `torch.no_grad()`: 測試時不需計算梯度，節省記憶體和計算時間。
  - `torch.max(outputs, 1)`: 找出輸出中概率最大的類別作為預測結果。
  - 計算總正確數和總樣本數，得出準確率。

#### **8. 訓練模型**
```python
for epoch in range(1):  # 只訓練 1 輪
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()  # 清空累積的梯度
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()        # 反向傳播計算梯度
        optimizer.step()       # 更新參數

        if i % TEST_FREQUENCY == 0:
            acc = test(net, test_loader)
            print(f"Epoch {epoch+1}, Batch {i}, Accuracy: {acc:.2f}%")
```
- **訓練步驟解析：**
  1. **清空梯度**：避免梯度累加影響當前批次。
  2. **前向傳播**：計算模型預測值。
  3. **計算損失**：比較預測值和真實標籤。
  4. **反向傳播**：根據損失計算各參數的梯度。
  5. **更新參數**：優化器根據梯度調整參數。
- **為什麼只訓練 1 個 Epoch？**  
  示例中為了快速演示，實際需多輪訓練（如 10 輪）才能達到高準確率。

#### **9. 保存模型並測試最終準確率**
```python
acc = test(net, test_loader)
print(f"Final Accuracy: {acc:.2f}%")
torch.save(net, 'model.pth')
```
- **保存模型目的**：訓練好的模型可重複使用，避免重新訓練。

#### **10. 預測並可視化結果**
```python
def predict_some():
    dataiter = iter(test_loader)
    imgs, labels = next(dataiter)
    with torch.no_grad():
        outputs = net(imgs[:6])
        _, predicted = torch.max(outputs, 1)
    # 畫出預測結果與真實標籤對比
    ...
predict_some()
```
- **結果解析：**  
  顯示 6 張測試圖片的預測結果（P）和真實標籤（T），直觀檢查模型表現。例如，若出現 `P:7 / T:1`，表示模型將數字 1 錯誤預測為 7。

### **總結思路**
1. **數據準備**：加載並預處理數據，確保格式正確。
2. **模型設計**：選擇合適的網絡結構（如 LeNet-5）提取特徵。
3. **訓練配置**：設定超參數、損失函數和優化器。
4. **迭代訓練**：通過多輪訓練逐步調整模型參數，降低預測誤差。
5. **評估與應用**：測試模型性能，保存並用於新數據預測。

### **常見問題**
- **為什麼準確率不高？**  
  可能原因：訓練輪數不足、學習率不當、模型結構過於簡單。可嘗試增加 Epoch、調整學習率或使用更複雜的網絡（如 ResNet）。
- **如何改進模型？**  
  - 數據增強：對訓練圖片做旋轉、平移等，增加多樣性。
  - 調整超參數：例如使用學習率衰減（Learning Rate Decay）。
  - 更深的網絡：例如增加卷積層或全連接層。
