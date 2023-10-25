import torch
import torch.nn as nn
import clip


def maml_init_(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module


class Lambda(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/misc.py)
    **Description**
    Utility class to create a wrapper based on a lambda function.
    **Arguments**
    * **lmb** (callable) - The function to call in the forward pass.
    **Example**
    ~~~python
    mean23 = Lambda(lambda x: x.mean(dim=[2, 3]))  # mean23 is a Module
    x = features(img)
    x = mean23(x)
    x = x.flatten()
    ~~~
    """

    def __init__(self, lmb):
        super(Lambda, self).__init__()
        self.lmb = lmb

    def forward(self, *args, **kwargs):
        return self.lmb(*args, **kwargs)


class ConvBlock(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 max_pool=True,
                 max_pool_factor=1.0,
                 bn_affine=True):
        super(ConvBlock, self).__init__()
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = torch.nn.MaxPool2d(
                kernel_size=stride,
                stride=stride,
                ceil_mode=False,
            )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = torch.nn.BatchNorm2d(
            out_channels,
            affine=bn_affine,
            # eps=1e-3,
            # momentum=0.999,
            # track_running_stats=False,
        )
        if bn_affine:
            torch.nn.init.uniform_(self.normalize.weight)
        self.relu = torch.nn.ReLU()

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1,
            bias=True,
        )
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(torch.nn.Sequential):

    # NOTE:
    #     Omniglot: hidden=64, channels=1, no max_pool
    #     MiniImagenet: hidden=32, channels=3, max_pool

    def __init__(self,
                 hidden=64,
                 channels=1,
                 max_pool=False,
                 layers=4,
                 max_pool_factor=1.0,
                 bn_affine=True):
        core = [ConvBlock(channels,
                          hidden,
                          (3, 3),
                          max_pool=max_pool,
                          max_pool_factor=max_pool_factor,
                          bn_affine=bn_affine),
                ]
        for _ in range(layers - 1):
            core.append(ConvBlock(hidden,
                                  hidden,
                                  kernel_size=(3, 3),
                                  max_pool=max_pool,
                                  max_pool_factor=max_pool_factor,
                                  bn_affine=bn_affine))
        super(ConvBase, self).__init__(*core)


class OmniglotCNN(torch.nn.Module):
    """
    [Source](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)
    **Description**
    The convolutional network commonly used for Omniglot, as described by Finn et al, 2017.
    This network assumes inputs of shapes (1, 28, 28).
    **References**
    1. Finn et al. 2017. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.” ICML.
    **Arguments**
    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.
    **Example**
    ~~~python
    model = OmniglotCNN(output_size=20, hidden_size=128, layers=3)
    ~~~
    """

    def __init__(self, output_size=5, hidden_size=64, layers=4, bn_affine=True):
        super(OmniglotCNN, self).__init__()
        self.hidden_size = hidden_size
        self.base = ConvBase(
             hidden=hidden_size,
             channels=1,
             max_pool=False,
             layers=layers,
             bn_affine=bn_affine,
        )
        self.features = torch.nn.Sequential(
            Lambda(lambda x: x.view(-1, 1, 28, 28)),
            self.base,
            Lambda(lambda x: x.mean(dim=[2, 3])), 
            nn.Flatten(),
        )
        self.full_conn = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.full_conn.weight.data.normal_()
        self.full_conn.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.full_conn(x)
        return x


class OmniglotDSVDD(nn.Module):
    def __init__(self, config=None):
        super(OmniglotDSVDD, self).__init__()

        output_size=config['output_size']
        hidden_size=config['hidden_size'] #64, 128
        layers=config['layers']

        self.logits = OmniglotCNN(output_size=output_size, 
                                  hidden_size=hidden_size, 
                                  layers=layers,
                                  bn_affine=config['bn_affine'])

        self.batch_norm = nn.BatchNorm1d(
            output_size,
            affine=config['bn_affine'],
            # eps=1e-3,
            # momentum=0.999,
            # track_running_stats=False,
        )
        if config['bn_affine']:
            nn.init.uniform_(self.batch_norm.weight)
        self.center = nn.Parameter(torch.ones(output_size))
        nn.init.normal_(self.center)

    def forward(self, x):
        x = self.logits(x)
        x = self.batch_norm(x)
        return x, self.center


class Cifar100CNN(torch.nn.Module):
    def __init__(self, output_size=5, hidden_size=64, layers=4, bn_affine=True):
        super(Cifar100CNN, self).__init__()
        self.hidden_size = hidden_size
        self.base = ConvBase(
             hidden=hidden_size,
             channels=3,
             max_pool=False,
             layers=layers,
             bn_affine=bn_affine,
        )
        self.features = torch.nn.Sequential(
            Lambda(lambda x: x.view(-1, 3, 32, 32)),
            self.base,
            Lambda(lambda x: x.mean(dim=[2, 3])),
            nn.Flatten(),
        )
        self.full_conn = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.full_conn.weight.data.normal_()
        self.full_conn.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.full_conn(x)
        return x


class Cifar100DSVDD(nn.Module):
    def __init__(self, config=None):
        super(Cifar100DSVDD, self).__init__()

        output_size=config['output_size']
        hidden_size=config['hidden_size'] #64, 128
        layers=config['layers']

        self.logits = Cifar100CNN(output_size=output_size, 
                                  hidden_size=hidden_size, 
                                  layers=layers,
                                  bn_affine=config['bn_affine'])

        self.batch_norm = nn.BatchNorm1d(
            output_size,
            affine=config['bn_affine'],
            # eps=1e-3,
            # momentum=0.999,
            # track_running_stats=False,
        )
        if config['bn_affine']:
            nn.init.uniform_(self.batch_norm.weight)
        self.center = nn.Parameter(torch.ones(output_size))
        nn.init.normal_(self.center)

    def forward(self, x):
        x = self.logits(x)
        x = self.batch_norm(x)
        return x, self.center


class CNN4Backbone(ConvBase):

    def __init__(
        self,
        hidden_size=64,
        layers=4,
        channels=3,
        max_pool=True,
        max_pool_factor=None,
        bn_affine=True
    ):
        if max_pool_factor is None:
            max_pool_factor = 4 // layers
        super(CNN4Backbone, self).__init__(
            hidden=hidden_size,
            layers=layers,
            channels=channels,
            max_pool=max_pool,
            max_pool_factor=max_pool_factor,
            bn_affine=bn_affine,
        )

    def forward(self, x):
        x = super(CNN4Backbone, self).forward(x)
        x = x.reshape(x.size(0), -1)
        return x


class CNN4(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)
    **Description**
    The convolutional network commonly used for MiniImagenet, as described by Ravi et Larochelle, 2017.
    This network assumes inputs of shapes (3, 84, 84).
    Instantiate `CNN4Backbone` if you only need the feature extractor.
    **References**
    1. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.
    **Arguments**
    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.
    * **channels** (int, *optional*, default=3) - The number of channels in input.
    * **max_pool** (bool, *optional*, default=True) - Whether ConvBlocks use max-pooling.
    * **embedding_size** (int, *optional*, default=None) - Size of feature embedding.
        Defaults to 25 * hidden_size (for mini-Imagenet).
    **Example**
    ~~~python
    model = CNN4(output_size=20, hidden_size=128, layers=3)
    ~~~
    """

    def __init__(
        self,
        output_size,
        hidden_size=64,
        layers=4,
        channels=3,
        max_pool=True,
        embedding_size=None,
        bn_affine=True,
    ):
        super(CNN4, self).__init__()
        if embedding_size is None:
            embedding_size = 25 * hidden_size
        self.features = CNN4Backbone(
            hidden_size=hidden_size,
            channels=channels,
            max_pool=max_pool,
            layers=layers,
            max_pool_factor=4 // layers,
            bn_affine=bn_affine,
        )
        # self.classifier = torch.nn.Linear(
        #     embedding_size,
        #     output_size,
        #     bias=True,
        # )
        # maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        # x = self.classifier(x)
        return x


class CNN4DSVDD(nn.Module):
    def __init__(self, config=None):
        super(CNN4DSVDD, self).__init__()

        output_size=config['output_size']
        hidden_size=config['hidden_size'] #64, 128
        layers=config['layers']

        self.logits = CNN4(output_size=output_size, 
                           hidden_size=hidden_size, 
                           layers=layers,
                           embedding_size=hidden_size*4, # hidden_size*4 for cifar100, hidden_size*25 for mini-ImageNet
                           bn_affine=config['bn_affine'])

        self.batch_norm = nn.BatchNorm1d(
            output_size,
            affine=config['bn_affine'],
            # eps=1e-3,
            # momentum=0.999,
            # track_running_stats=False,
        )
        if config['bn_affine']:
            nn.init.uniform_(self.batch_norm.weight)
        self.center = nn.Parameter(torch.ones(output_size))
        nn.init.normal_(self.center)

    def forward(self, x):
        x = self.logits(x)
        x = self.batch_norm(x)
        return x, self.center

class CNN4ClS(nn.Module):
    def __init__(self, config=None):
        super(CNN4ClS, self).__init__()

        output_size=config['output_size']
        hidden_size=config['hidden_size'] #64, 128
        layers=config['layers']
        self.embedding_size = hidden_size*4

        self.logits = CNN4(output_size=output_size, 
                           hidden_size=hidden_size, 
                           layers=layers,
                           channels= config['channel_size'],
                           embedding_size=hidden_size*4, # hidden_size*4 for cifar100, hidden_size*25 for mini-ImageNet
                           bn_affine=config['bn_affine'])

        self.classifier = torch.nn.Linear(
            self.embedding_size,
            output_size,
            bias=True,
        )

    def forward(self, x):
        z = self.logits(x)
        logit = self.classifier(z)
        return logit,z

def CLIP(model_path):
    model,data_transform = clip.load(model_path, 'cuda',jit=False)
    return model,data_transform