import torch
import torch.nn as nn


class OutcomeClassifier(nn.Module):
    """
    Deep Neural Network for Court Case Outcome Prediction.
    Input: Sentence-BERT embeddings
    Output: Case outcome classes
    """

    def __init__(self, input_dim=384, hidden_dim=128, num_classes=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)