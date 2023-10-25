import torch
import torch.nn as nn

class FeatureSVDD(nn.Module):
    def __init__(self, config=None):
        super(FeatureSVDD, self).__init__()

        x_dim = config['x_dim']
        output_size=config['output_size']
        layers=config['layers']

        enc = []
        input_dim = x_dim
        for _ in range(layers):
            enc.append(nn.Linear(input_dim, int(input_dim/2)))
            enc.append(nn.BatchNorm1d(int(input_dim/2), affine=config['bn_affine']))
            enc.append(nn.ReLU())
            input_dim = int(input_dim/2)
        self.enc = nn.Sequential(*enc)
        self.fc = nn.Linear(input_dim, output_size)
        self.batch_norm = nn.BatchNorm1d(
            output_size,
            affine=config['bn_affine'],
        )
        if config['bn_affine']:
            nn.init.uniform_(self.batch_norm.weight)

        self.center = nn.Parameter(torch.ones(output_size))
        nn.init.normal_(self.center)

    def forward(self, x):
        z = self.enc(x)
        z = self.fc(z)
        z = self.batch_norm(z)
        return z, self.center