from datetime import datetime
import yaml
import shutil
import os
import argparse
import torch

from classifier.multiple_instance_learner import MILLearner
from models.efficientnet import efficientnet
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--debug', type=bool, default=False)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def get_model(config, num_classes, checkpoint_path=None, **kwargs) -> torch.nn.Module:
    if config['type'] == 'efficientnet_pytorch':
        model = efficientnet(num_classes, **config['params'], **kwargs)
    else:
        raise ValueError('Invalid model specified: {}'.format(config['type']))

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model


def get_learner(learner_type, model, config, fold, dt_string, debug):
    if learner_type == 'mil':
        return MILLearner(model, config, fold, dt_string, debug)
    else:
        raise ValueError('Invalid learner specified: {}'.format(learner_type))


if __name__ == '__main__':
    args = parser.parse_args()
    config = load_config(args.config)
    classes = config['classes']
    num_classes = len(classes)
    model_name = config['model_name']

    model = get_model(config['network'], num_classes)
    dt_string = datetime.now().strftime('%Y%m%d_%H%M')
    
    working_dir = os.path.join('runs', config['model_name'], dt_string)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        shutil.copy(args.config, os.path.join(working_dir, 'config.yaml'))

    for fold in config['folds']:
        logger = TensorBoardLogger('tensorboard/{}/{}'.format(dt_string, fold), name=model_name)
        learner = get_learner(config['learner_type'], model, config, fold, dt_string, args.debug)
        trainer = Trainer(val_check_interval=config['val_every'],
                          max_epochs=config['epochs'],
                          logger=logger,
                          nb_sanity_val_steps=1,
                          gpus=args.gpus)

        trainer.fit(learner)