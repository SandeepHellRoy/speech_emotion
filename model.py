import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNNBiLSTMAttention(nn.Module):
    def __init__(self, n_classes):
        super(EmotionCNNBiLSTMAttention, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),  # [B, 1, 128, T]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),  # ✅ Dropout after BN
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),  # ✅ Dropout again
            nn.MaxPool2d((2, 2))
        )

        # LSTM input: (B, T, 64*32)
        self.lstm = nn.LSTM(input_size=64*32, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(0.2)  # ✅ Dropout after LSTM

        self.attn = nn.Linear(256, 1)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.cnn(x)  # [B, C, H, W]
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, -1)  # [B, T, C*H]
        
        x, _ = self.lstm(x)  # [B, T, 256]
        x = self.dropout_lstm(x)

        attn_weights = torch.softmax(self.attn(x), dim=1)  # [B, T, 1]
        x = torch.sum(attn_weights * x, dim=1)  # Weighted sum → [B, 256]

        out = self.fc(x)  # [B, n_classes]
        return out
