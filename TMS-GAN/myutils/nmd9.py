import torch.nn as nn

class NMD(nn.Module):
    def __init__(self, cls_dim=3):
        super().__init__()
        
        self.channel = 3
        self.n_feats = 64

        self.conv1_1 = nn.Conv2d(self.channel,     self.n_feats * 1, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(self.n_feats * 1, self.n_feats * 1, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(self.n_feats * 1, self.n_feats * 2, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(self.n_feats * 2, self.n_feats * 2, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(self.n_feats * 2, self.n_feats * 4, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(self.n_feats * 4, self.n_feats * 4, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(self.n_feats * 4, self.n_feats * 8, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(self.n_feats * 8, self.n_feats * 8, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(self.n_feats * 8, cls_dim, kernel_size=3, padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.act = nn.ReLU()

        self._weight_initialize()

    def _weight_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.act(x)
        x = self.conv1_2(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.conv2_1(x)
        x = self.act(x)
        x = self.conv2_2(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.conv3_1(x)
        x = self.act(x)
        x = self.conv3_2(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.conv4_1(x)
        x = self.act(x)
        x = self.conv4_2(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.conv5_1(x)
        x = self.gap(x).view(x.size(0), x.size(1))
        return x


