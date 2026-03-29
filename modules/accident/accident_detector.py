import torch
import torch.nn as nn
import torchvision.models as models


class AccidentCNNLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, num_classes=2):
        super(AccidentCNNLSTM, self).__init__()

        # CNN feature extractor — use pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        # Remove final classification layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_output_size = 512

        # LSTM for temporal sequence learning
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, sequence_length, C, H, W)
        batch_size, seq_len, C, H, W = x.shape

        # Pass each frame through CNN
        cnn_features = []
        for t in range(seq_len):
            feat = self.cnn(x[:, t, :, :, :])
            feat = feat.view(batch_size, -1)
            cnn_features.append(feat)

        # Stack into sequence
        cnn_out = torch.stack(cnn_features, dim=1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_out)

        # Take last timestep output
        out = self.classifier(lstm_out[:, -1, :])
        return out