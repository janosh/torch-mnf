from torch import nn


class LeNet(nn.Sequential):
    """Just your regular LeNet."""

    def __init__(self, **kwargs):
        layers = [
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=-1),
        ]
        super().__init__(*layers, **kwargs)

    def forward(self, x):
        for lyr in self:
            x = lyr(x)
        return x
