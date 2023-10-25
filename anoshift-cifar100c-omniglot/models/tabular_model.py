import torch
import torch.nn as nn
import clip
import numpy as np

class TabNN(torch.nn.Module):
    """

    """

    def __init__(self, input_size,output_size=5, hidden_size=64, layers=4, bn_affine=True,bias=True):
        super(TabNN, self).__init__()
        self.hidden_size = hidden_size

        enc = []
        for _ in range(layers - 1):
            # hidden_size = int(input_dim/2)
            enc.append(nn.Linear(input_size, hidden_size,bias=bias))
            if bn_affine:
                enc.append(nn.BatchNorm1d(hidden_size,affine=bn_affine))
            enc.append(nn.ReLU())
            input_size = hidden_size

        self.enc = nn.Sequential(*enc)
        self.fc = nn.Linear(input_size, output_size,bias=bias)

    def forward(self, x):
        x = self.enc(x)
        x = self.fc(x)
        return x


class TabDSVDD(nn.Module):
    def __init__(self, config=None):
        super(TabDSVDD, self).__init__()
        input_size = config['input_size']
        output_size=config['output_size']
        hidden_size=config['hidden_size'] #64, 128
        layers=config['layers']
        self.device = config['device']
        self.logits = TabNN(input_size,
                            output_size=output_size, 
                            hidden_size=hidden_size, 
                            layers=layers,
                            bn_affine=True)


        if config['bn_affine']:
            self.batch_norm = nn.BatchNorm1d(
                output_size,
                affine=config['bn_affine'],
                # eps=1e-3,
                # momentum=0.999,
                # track_running_stats=False,
            )
            nn.init.uniform_(self.batch_norm.weight)
        self.center = nn.Parameter(torch.ones(output_size))
        nn.init.normal_(self.center)
        self.bn = config['bn_affine']
    def forward(self, x):
        x = x.type(torch.FloatTensor).to(self.device)
        x = self.logits(x)
        if self.bn:
            x = self.batch_norm(x)
        return x, self.center
    

        
class TabTransformNet(nn.Module):
    def __init__(self, x_dim,h_dim,bias,num_layers):
        super(TabTransformNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers-1):
            net.append(nn.Linear(input_dim,h_dim,bias=bias))
            # net.append(nn.BatchNorm1d(h_dim,affine=bias))
            net.append(nn.ReLU())
            input_dim= h_dim
        net.append(nn.Linear(input_dim,x_dim,bias=bias))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)

        return out



def _make_nets(x_dim,config):

    zdim = config['output_size']
    hdim = config['hidden_size']
    trans_dim = config['hidden_size']

    trans_nlayers = 3
    num_trans = 19

    enc = TabNN(x_dim,output_size=zdim, 
                hidden_size=hdim, 
                layers=config['layers'],
                bn_affine=config['bn_affine'])
    trans = nn.ModuleList(
        [TabTransformNet(x_dim, trans_dim, True, trans_nlayers) for _ in range(num_trans)])

    return enc,trans

class TabNeutralAD(nn.Module):
    def __init__(self, config):
        super(TabNeutralAD, self).__init__()
        x_dim = config['input_size']
        self.enc,self.trans = _make_nets(x_dim,config)
        self.num_trans = 19
        self.trans_type = "forward"
        self.device = config['device']

        self.z_dim = config['output_size']
        self.batch_norm = nn.BatchNorm1d(
            self.z_dim,
            affine=config['bn_affine'],
            # eps=1e-3,
            # momentum=0.999,
            # track_running_stats=False,
        )
        if config['bn_affine']:
            nn.init.uniform_(self.batch_norm.weight)
        self.center = nn.Parameter(torch.ones(1,self.z_dim))
        nn.init.normal_(self.center)
#        weights_init(self.trans)
    def forward(self,x):
        x = x.type(torch.FloatTensor).to(self.device)

        x_T = torch.empty(x.shape[0],self.num_trans,x.shape[-1]).to(x)
        for i in range(self.num_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_T[:, i] = mask
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
            elif self.trans_type == 'mul':
                x_T[:, i] = torch.sigmoid(mask) * x
        x_cat = torch.cat([x.unsqueeze(1),x_T],1)
        zs = self.enc(x_cat.reshape(-1,x.shape[-1]))
        # zs = zs.reshape(x.shape[0],-1)
        # zs = self.batch_norm(zs)
        zs = zs.reshape(x.shape[0],self.num_trans+1,self.z_dim)
        # zs[:,0]= zs[:,0] + self.center
        return zs,self.center

