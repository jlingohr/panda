import yaml
import torch
import sys
import albumentations as albu


def get_criterion(config):
    return torch.nn.CrossEntropyLoss()


def parse_transform(trans_type, trans_params):
    """
    Returns a single transform from a configuration

    :param trans_type: String with the name of the transform
    :param trans_params: Params to pass to that transform's constructor
    :return: A single transform object
    """
    if hasattr(sys.modules[__name__], trans_type):
        return getattr(sys.modules[__name__], trans_type)(**trans_params)
    else:
        return getattr(albu, trans_type)(**trans_params)


def create_transforms(transform_list):
    """
    Parses and composes a list of transform configurations.

    :param transform_list:
    :return: A list of transforms that has been torchvision.transform.Compose'd
    """

    trans_list = []
    for trans in transform_list:
        (trans_type, trans_params), = trans.items()
        trans_list.append(parse_transform(trans_type, trans_params))

    return albu.Compose(trans_list)