import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import model_zoo
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import (
    get_same_padding_conv2d,
    round_filters,
    get_model_params
)

url_map = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
}

# train with Adversarial Examples(AdvProp)
# check more details in paper(Adversarial Examples Improve Image Recognition)
url_map_advprop = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth',
}


def load_pretrained_weights(model, model_name, weights_path=None, load_fc=True, advprop=False):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): The whole model of efficientnet.
        model_name (str): Model name of efficientnet.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        advprop (bool): Whether to load pretrained weights
                        trained with advprop (valid when weights_path is None).
    """
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)
    else:
        # AutoAugment or Advprop (different preprocessing)
        url_map_ = url_map_advprop if advprop else url_map
        state_dict = model_zoo.load_url(url_map_[model_name])

    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
    #         assert not ret.missing_keys, f'Missing keys when loading pretrained weights: {ret.missing_keys}'
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
    #         assert set(ret.missing_keys) == set(
    #             ['_fc.weight', '_fc.bias']), f'Missing keys when loading pretrained weights: {ret.missing_keys}'
    #     assert not ret.unexpected_keys, f'Missing keys when loading pretrained weights: {ret.unexpected_keys}'

    print('Loaded pretrained weights for {}'.format(model_name))


def efficientnet(num_classes, model_name, drop_connect_rate=0.8, dropout=0.5, advprop=False, **kwargs):
    model = EfficientNetPooled.from_name(model_name, num_classes=num_classes, drop_connect_rate=drop_connect_rate,
                                         dropout_rate=dropout)
    load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
    return model


class Mish(nn.Module):

    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input * torch.tanh(F.softplus(input))


class EfficientNetPooled(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super(EfficientNetPooled, self).__init__(blocks_args, global_params)
        self._max_pooling = nn.AdaptiveMaxPool2d(1)
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * self._fc.in_features, self._fc.in_features),
            Mish(),
            nn.BatchNorm1d(self._fc.in_features),
            self._dropout,
            self._fc
        )

    def forward(self, inputs):
        bs, k, c, w, h = inputs.shape

        x = inputs.view(-1, c, w, h)

        # bs*N x C x W x H
        # Convolution layers
        x = self.extract_features(x)  # bs*N x C x W' x H'
        shape = x.shape

        # Concatenate the output for tiles into a single map
        x = x.view(-1, k, shape[1], shape[2], shape[3]).permute(0, 2, 1, 3,
                                                                4).contiguous()  # hardcoding 16 patches/image
        x = x.view(-1, shape[1], shape[2] * k, shape[3])

        x_avg = self._avg_pooling(x)
        x_max = self._max_pooling(x)
        x = torch.cat([x_avg, x_max], dim=1)

        x = self.out(x)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.
        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
