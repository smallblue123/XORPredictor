import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from PIL import Image


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 權重初始化
        self.W1 = np.random.randn(self.input_size, self.hidden_size).astype(np.float64)
        self.b1 = np.random.randn(1, self.hidden_size).astype(np.float64)
        self.W2 = np.random.randn(self.hidden_size, self.output_size).astype(np.float64)
        # self.b2 = np.random.randn(1, self.output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tangent_sigmoid(self.z1)
        # print(self.a1)
        self.z2 = np.dot(self.a1, self.W2)
        self.a2 = self.z2  # 對於回歸問題，不使用激活函數
        return self.a2

    def backward(self, X, y, output, learning_rate):
        m = y.shape[0]

        # 計算輸出層誤差
        error_output = output - y.reshape(-1, 1)

        # 計算輸出層梯度
        dW2 = np.dot(self.a1.T, error_output) / m

        # 計算隱藏層誤差
        error_hidden = np.dot(error_output, self.W2.T) * self.tangent_sigmoid_derivative(self.z1)

        # 計算隱藏層梯度
        dW1 = np.dot(X.T, error_hidden) / m
        db1 = np.sum(error_hidden, axis=0, keepdims=True) / m

        # 更新權重
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2

    def tangent_sigmoid(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def tangent_sigmoid_derivative(self, x):
        return 1-self.tangent_sigmoid(x)**2

    def train(self, X, y, epochs, learning_rate):
        loss = []
        frames = []

        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            # if epoch % 100 == 99:
            loss.append(np.mean((output - y.reshape(-1, 1)) ** 2)/2)
            print(f'Epoch {epoch}, Loss: {loss[-1]}')

            if epoch % 1000 == 999:
                fig, ax = plt.subplots()
                self.plot_decision_boundary(ax, X, y, epoch)

                plt.savefig(f'decision_boundary_epoch_{epoch}.png')
                plt.close()


        # plt.show()
        return loss

    def predict(self, X):
        return self.forward(X)

    def plot_decision_boundary(self, ax, X, Y, epoch):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = (Z > 0.5).astype(int)  # 將輸出轉換為0和1
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=Y.flatten(), edgecolors='k', marker='o')
        ax.set_title(f'Epoch {epoch}')
        # plt.show()
        # 保存圖片

        x1_1 = np.linspace(x_min, x_max, 100)
        x2_1 = -1 * (self.W1[0, 0] * x1_1 + self.b1[0, 0]) / self.W1[1, 0]
        x1_2 = np.linspace(x_min, x_max, 100)
        x2_2 = -1 * (self.W1[0, 1] * x1_2 + self.b1[0, 1]) / self.W1[1, 1]

        # 畫y1線
        ax.plot(x1_1, x2_1, label='y1', color='r')
        ax.legend()
        # 畫y2線
        ax.plot(x1_2, x2_2, label='y2', color='b')
        ax.legend()

        # 設定範圍
        plt.xlim(-1, 2)
        plt.ylim(-1, 2)


def generate_xor_data(num_samples):
    X = np.empty((num_samples,2))
    Y = np.empty(num_samples)
    random_test = np.random.random(num_samples)
    for i in range(len(random_test)):
        if random_test[i]<0.25:
            x1 = np.random.uniform(-0.5, 0.2)
            x2 = np.random.uniform(-0.5, 0.2)
            y = 0
        elif random_test[i]<0.5:
            x1 = np.random.uniform(-0.5, 0.2)
            x2 = np.random.uniform(0.8, 1.5)
            y = 1
        elif random_test[i] < 0.75:
            x1 = np.random.uniform(0.8, 1.5)
            x2 = np.random.uniform(-0.5, 0.2)
            y = 1
        else:
            x1 = np.random.uniform(0.8, 1.5)
            x2 = np.random.uniform(0.8, 1.5)
            y = 0
        X[i] = [x1, x2]
        Y[i] = y
    return X, Y


def find_best_model(X_train_xor, y_train_xor, X_test_xor, n=30):
    # 模擬建模參數
    num_models = 30
    models = []
    best_model_results = []
    losses = []
    best_model = None
    best_loss = [math.inf]
    best_animation = []

    # 神經元數
    input_size = 2
    hidden_size = 2
    output_size = 1

    for i in range(n):
        nn = NeuralNetwork(input_size, hidden_size, output_size)
        # 訓練參數
        epochs = 30000
        learning_rate = 0.005
        loss = nn.train(X_train_xor, y_train_xor, epochs, learning_rate)
        model_results = nn.predict(X_test_xor)

        if loss[-1] < best_loss[-1]:
            best_loss = loss
            best_model = nn
            best_model_results = model_results

        models.append(nn)
        losses.append(loss)
    Boxplot(losses)
    return best_model


def Boxplot(losses):
    # 盒須圖
    plt.figure(figsize=(8, 6))
    plt.boxplot(np.array(losses)[:, -1])
    plt.xlabel('')
    plt.ylabel('loss')
    plt.title('Boxplot of Model Results with Same Number of Hidden Layer Neurons')
    plt.show()


# 生成 XOR 資料集
X, Y = generate_xor_data(10000)

# 以 8:2 的比例分成訓練和測試資料集
train_size_xor = int(0.8 * 10000)
X_train_xor, X_test_xor = X[:train_size_xor], X[train_size_xor:]
y_train_xor, y_test_xor = Y[:train_size_xor], Y[train_size_xor:]


# find_best_model(X_train_xor, y_train_xor, X_test_xor)


# 神經元數
input_size = 2
hidden_size = 2
output_size = 1

# for i in range(30):
nn = NeuralNetwork(input_size, hidden_size, output_size)

# 訓練參數
epochs = 30000
learning_rate = 0.005

loss = nn.train(X_train_xor, y_train_xor, epochs, learning_rate)

model_results = nn.predict(X_test_xor)

# 最佳隱藏層數量下的真實與預測解比較
plt.figure(figsize=(8, 6))
plt.scatter(y_test_xor, model_results, s=10)
# plt.scatter(y_test_xor, y_test_xor, label='real', s=10)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Real vs Predict')
plt.legend()
plt.savefig(f'Real vs Predict.png')
plt.show()


# 最佳隱藏層數量下的學習曲線
plt.figure(figsize=(8, 6))
plt.plot(range(len(loss)), loss)
plt.xlabel('Turn')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.savefig(f'Learning Curve.png')
plt.show()


# 最佳隱藏層數量下的誤差直方圖
test_errors = model_results.flatten() - y_test_xor
train_errors = nn.predict(X_train_xor).flatten() - y_train_xor

plt.figure(figsize=(8, 6))
plt.hist(np.array(test_errors).flatten(), bins=100, alpha=1, label='Test Errors')
plt.hist(np.array(train_errors).flatten(), bins=100, alpha=0.8, label='Train Errors')
plt.legend()
plt.xlabel('Error')
plt.ylabel('times')
plt.title('Error Histogram')
plt.savefig(f'Error Histogram.png')
plt.show()
