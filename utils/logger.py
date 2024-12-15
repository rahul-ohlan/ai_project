import os
import torch
import torch.nn as nn
import yaml

def save_config(config, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(config, f)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class EarlyStopping:
    def __init__(self, save_path, save_frequency, patience=7, verbose=True, delta=0.00001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.save_path = save_path
        self.save_frequency = save_frequency

    def __call__(self, val_loss, train_loss, model, epoch, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, train_loss, model, epoch, optimizer)
        elif score < self.best_score + self.delta: # if val_loss is not decreasing more than delta
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, train_loss, model, epoch, optimizer)
            self.counter = 0
        
        # save at a frequency if provided
        if self.save_frequency > 0 and epoch % self.save_frequency == 0:
            self.save_epoch_checkpoint(val_loss, train_loss, model, epoch, optimizer)

    def save_checkpoint(self, val_loss, train_loss, model, epoch, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # check for distributed training
        model_state_dict = model.module.state_dict() if isinstance(model,(nn.DataParallel, nn.parallel.DistributedDataParallel)) else model.state_dict()
        
        chkpt = os.path.join(self.save_path, f"best_val.pt")

        torch.save({'epoch': epoch, 
                    "state_dict": model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss':train_loss,
                    'val_loss': val_loss}, chkpt)  # save checkpoint
        self.val_loss_min = val_loss

    def save_epoch_checkpoint(self, val_loss, train_loss, model, epoch, optimizer):
        '''Saves model at the specified epoch interval.'''
        if self.verbose:
            print(f'Saving model at epoch {epoch}...')

        # Check for distributed training
        model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

        chkpt = os.path.join(self.save_path, f"checkpoint_epoch_{epoch}.pt")

        torch.save({
            'epoch': epoch,
            "state_dict": model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, chkpt)  # Save model checkpoint at the specified frequency