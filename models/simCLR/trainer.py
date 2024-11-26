import os

import torch
import torch.distributed as dist

import wandb

from models.simCLR.modules.lars import LARS
from utils.logger import EarlyStopping

class simCLRTrainer:

    def __init__(self, config, rank):

        self.config = config
        self.rank = rank
        self.config['global_step'] = 0
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        loss_epoch = 0
        model.train()
        
        for step, data in enumerate(train_loader):
            optimizer.zero_grad()
            x_i = data['x_i'].cuda(non_blocking=True)
            x_j = data['x_i'].cuda(non_blocking=True)

            # positive pair, with encoding
            h_i, h_j, z_i, z_j = model(x_i, x_j)

            loss = criterion(z_i, z_j)
            loss.backward()
            torch.cuda.synchronize()

            optimizer.step()

            if dist.is_available() and dist.is_initialized():
                avg_loss = loss.data.clone()
                dist.all_reduce(avg_loss.div_(dist.get_world_size()))

            if self.rank == 0 and step % 50 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {avg_loss.item()}")

            
            # replace with wandb
            if self.rank == 0:
                # wandb.log({"Loss/train_epoch": avg_loss.item()}, step=self.config['global_step'])
                self.config['global_step'] += 1

            loss_epoch += avg_loss.item()
        return loss_epoch
    
    def _train(self, model, train_loader, optimizer, criterion, scheduler=None):


        USE_WANDB = self.config['use_wandb']
        if USE_WANDB and self.rank==0:
            wandb.init(project=self.config['wandb_project'], 
                       name=f"{self.config['experiment_ID']}",
                       entity=self.config['wandb_entity'])
            wandb.config.update(self.config)

        num_epochs = self.config['num_epochs']


        print('starting training...')

        val_checkpoint = os.path.join(self.config['checkpoint_dir'], self.config['model_chkpt_dir'], self.config['experiment_ID'])
        patience, delta = self.config['patience'], self.config['delta']
        early_stopping = EarlyStopping(save_path=val_checkpoint, save_frequency= self.config["save_every"], patience=patience, delta=delta)
        
        for epoch in range(num_epochs):

            # train epoch loss
            if dist.is_available() and dist.is_initialized():
                train_loader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss}")

            if USE_WANDB and self.rank==0:
                wandb.log({"train_loss": train_loss}, step = epoch)
            
            
            # early_stopping to save model checkpoint
            if self.rank == 0:
                early_stopping(val_loss=train_loss, train_loss=0, model=model, epoch=epoch, optimizer=optimizer)

                if early_stopping.early_stop: 
                    print("Early stopping")
                    break
            
            if scheduler is not None:
                scheduler.step()


    def fit(self, model, train_loader, optimizer, criterion, scheduler):
        self._train(model, train_loader, optimizer, criterion, scheduler)
        dist.destroy_process_group() # clean up
        return model
    

    
    def save_model(self, path, model):
        out = os.path.join(self.config['model_path'], "checkpoint_{}.tar".format(self.config['current_epoch']))

        # To save a DataParallel model generically, save the model.module.state_dict().
        # This way, you have the flexibility to load the model any way you want to any device you want.
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), out)
        else:
            torch.save(model.state_dict(), out)
