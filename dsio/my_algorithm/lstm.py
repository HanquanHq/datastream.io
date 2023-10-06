import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import percentileofscore


class MyDLAnomalyDetector:

    def __init__(
            self,
            window_size=300,
            threshold=0.99,
            lstm_hidden_dim=64,
            num_layers=1
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers

        # Initialize the LSTM model
        self.model = LSTMAnomalyDetector(input_size=1, hidden_dim=self.lstm_hidden_dim, num_layers=self.num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Initialize the sample buffer
        self.sample_buffer = torch.zeros(window_size, dtype=torch.float32)
        self.sample_index = 0

    def fit(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if len(x) >= self.window_size:
            self.sample_buffer = x[-self.window_size:]
        else:
            self.sample_buffer[:len(x)] = x
            self.sample_index = len(x)

    def update(self, x):
        x = torch.tensor([x], dtype=torch.float32)  # 将数据点包装成一维数组
        if self.sample_index + len(x) >= self.window_size:
            self.sample_buffer[:-len(x)] = self.sample_buffer[len(x):].clone()  # 克隆输入张量以避免内存冲突
            self.sample_buffer[-len(x):] = x
            self.sample_index = self.window_size
        else:
            self.sample_buffer[self.sample_index:self.sample_index + len(x)] = x
            self.sample_index += len(x)

    def score_anomaly(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        scores = []

        with torch.no_grad():
            for i in range(len(x)):
                input_data = self.sample_buffer[-self.window_size:].view(1, -1, 1)
                output = self.model(input_data)
                mse = self.criterion(output, x[i].view(1, -1))
                scores.append(mse.item())

        return scores

    def flag_anomaly(self, x):
        scores = self.score_anomaly(x)
        flags = [score > self.threshold for score in scores]
        return flags


class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers):
        super(LSTMAnomalyDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    # 设置阈值
    THRESHOLD = 0.99

    # 创建模拟数据
    np.random.seed(0)
    n_samples = 1000
    normal_data = np.random.normal(0, 1, n_samples)  # 正常数据
    anomaly_data = np.random.normal(5, 1, n_samples)  # 异常数据

    # 初始化实时异常检测器
    detector = MyDLAnomalyDetector(window_size=100, threshold=THRESHOLD)

    # 模拟数据流并检测异常
    all_scores = []
    all_data = []  # 收集模拟数据
    for i in range(2 * n_samples):
        if i < n_samples:
            data_point = normal_data[i]
        else:
            data_point = anomaly_data[i - n_samples]

        print(data_point)

        detector.update(data_point)
        scores = detector.score_anomaly([data_point])
        print(scores)
        all_scores.append(scores[0])

        all_data.append(data_point)  # 收集模拟数据

    import matplotlib.pyplot as plt

    # 绘制异常分数和阈值的图形
    plt.figure(figsize=(12, 6))
    plt.plot(all_scores, label='Anomaly Scores', color='blue')
    threshold_line = [detector.threshold] * len(all_scores)
    plt.plot(threshold_line, label='Threshold', linestyle='--', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.title('Real-Time Anomaly Detection')

    # 绘制模拟数据的图形
    plt.figure(figsize=(12, 6))
    plt.plot(all_data, label='Data', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Data Value')
    plt.legend()
    plt.title('Simulated Data')

    plt.show()
