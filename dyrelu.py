class DynamicReLU(nn.Module):
    def __init__(self, channels, type='B', K=2, ratio=8):
        super(DynamicReLU, self).__init__()
        self.channels = channels
        self.K = K
        self.ratio = ratio
        self.lambdas = torch.Tensor([1.] * K + [0.5]*K).float().to(device)
        self.init_v = torch.Tensor([1.] + [0.]*(2*K-1)).float().to(device)
        if type=='B':
            self.mid_channels = 2*K*channels
        elif type=='A':
            self.mid_channels = 2*K

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dynamic = nn.Sequential(
            nn.Linear(channels, channels//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels//ratio, self.mid_channels),
            nn.Sigmoid(),
            )

    def forward(self, x):
        x_temp = x.clone()
        b, c, _, _ = x.size()
        x = self.avg_pool(x).view(b, c)
        x = self.dynamic(x)
        x = 2 * x - 1
        if type=='B':
            relu_coefs = x.view(-1, self.channels, 2*self.K)*self.lambdas+self.init_v
            x_perm = x_temp.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.K] + relu_coefs[:, :, self.K:]
            output = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        elif type=='A':
            relu_coefs = x.view(-1, 2*self.K)*self.lambdas+self.init_v
            x_perm = x.transpose(0, -1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :self.K] + relu_coefs[:, self.K:]
            output = torch.max(output, dim=-1)[0].transpose(0,-1)
        return output