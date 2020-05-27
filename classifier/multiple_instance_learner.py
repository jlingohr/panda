from dataset.patches_dataset import PatchesDataset
from classifier.common import get_criterion, create_transforms

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    cohen_kappa_score
)

import pytorch_lightning as pl


class MILLearner(pl.LightningModule):
    def __init__(self, model, config, fold=0, dt_string=None, debug=False):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = get_criterion(config['criteria'])
        self.best_loss = np.inf
        self.data_root = config['data_root']
        self.fold = fold
        self.save_dir = '{}/{}'.format(dt_string, fold)
        self.debug = debug

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']

        logits = self.model(images)
        loss = self.criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        accuracy = torch.tensor(preds.eq(labels).sum().item() / (len(labels) * 1.))

        logger_logs = {'training_loss': loss,
                       'training_acc': accuracy}

        out = {
            'loss': loss,
            'progress_bar': {'training_loss': loss},
            'log': logger_logs
        }

        return out

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']

        logits = self.forward(images)
        loss = self.criterion(logits, labels)

        probs = torch.softmax(logits.detach(), dim=1)
        preds = torch.argmax(probs, dim=1)
        accuracy = torch.tensor(preds.eq(labels).sum().item() / (len(labels) * 1.))

        logger_logs = {'validation_loss': loss,
                       'validation_acc': accuracy}

        out = {
            'val_loss': loss,
            'val_acc': accuracy,
            'val_labels': labels,
            'val_preds': preds,
            'log': logger_logs
        }

        return out

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        results = {
            'log': {
                'training_loss': avg_loss
            },
            'progress_bar': {
                'training_loss': avg_loss
            }
        }

        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.stack([x['val_acc'] for x in outputs]).sum() / len(outputs)
        labels = [x['val_labels'].cpu().numpy() for x in outputs]
        labels = np.concatenate(labels, axis=0)
        preds = [x['val_preds'].cpu().numpy() for x in outputs]
        preds = np.concatenate(preds, axis=0)

        kappa_score = cohen_kappa_score(labels, preds, weights='quadratic')

        results = {
            'log': {
                'validation_loss': avg_loss,
                'validation_accuracy': acc,
                'validation_kappa_score': torch.tensor(kappa_score)
            },
            'progress_bar': {
                'validation_loss': avg_loss,
                'validation_accuracy': acc
            }
        }

        if avg_loss.item() < self.best_loss:
            self.best_loss = avg_loss.item()
            torch.save({
                'model_state_dict': self.model.state_dict(),
            }, '{}/best.pth'.format(self.save_dir))
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, '{}/last.pth'.format(self.save_dir))

        return results

    def prepare_data(self):
        train_transforms = create_transforms(self.config['transforms']['train'])
        val_transforms = create_transforms(self.config['transforms']['val'])

        df = pd.read_csv(os.path.join(self.data_root, self.config['csv_name']))
        crop_df = pd.read_csv(os.path.join(self.data_root, self.config['crop_csv']))
        train_indices = np.load(os.path.join(self.data_root, 'train_{}.npy'.format(self.fold)))
        val_indices = np.load(os.path.join(self.data_root, 'val_{}.npy'.format(self.fold)))

        train_df = df[df.index.isin(train_indices)].reset_index()
        val_df = df[df.index.isin(val_indices)].reset_index()

        if self.debug:
            train_df = train_df[:1000]
            val_df = val_df[:1000]

        train_crops = crop_df[crop_df.image_id.isin(train_df.image_id)]
        val_crops = crop_df[crop_df.image_id.isin(val_df.image_id)]

        self.train_dataset = PatchesDataset(self.data_root, train_df, train_crops, train_transforms, window_size=self.config['window_size'])
        self.val_dataset = PatchesDataset(self.data_root, val_df, val_crops, val_transforms, window_size=self.config['window_size'])

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset,
                                batch_size=self.config['batch_size'],
                                shuffle=True,
                                drop_last=True,
                                num_workers=4)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset,
                                batch_size=self.config['batch_size'],
                                shuffle=False,
                                drop_last=False,
                                num_workers=4)
        return dataloader

    def configure_optimizers(self):
        optimizer = getattr(torch.optim,
                            self.config['optimizer']['type'])(self.model.parameters(),
                                                              **self.config['optimizer']['params'])
        scheduler = getattr(torch.optim.lr_scheduler,
                            self.config['scheduler']['type'])(optimizer,
                                                              **self.config['scheduler']['params'])
        scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'monitor': 'validation_loss'
        }

        return [optimizer], [scheduler]