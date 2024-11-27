import torch
import torch.nn as nn
from utils.logger import AvgMeter, EarlyStopping
import wandb
import os

class CLIPTrainer:

    def __init__(self, config):
        self.config = config

    def train_epoch(self,model, dataloader, device, optimizer, scheduler=None):

        model.train()
        train_loss = AvgMeter()

        for batch_idx, batch in enumerate(dataloader):

            image = batch["image"].to(device) # gene expression
            mol_embed = batch["smiles_emb"].to(device)
            smiles = batch["smiles"]
            dosage = batch["dosage"].to(device)
            


            # get model outputs
            if(self.config['add_dosage']):
                gene_features, mol_features, logit = model(image, mol_embed, dosage) # it returns unnormed embeddings
            else:
                gene_features, mol_features, logit = model(image, mol_embed, None)
            # print(image_features.size(), text_features.size())

            # compute loss
            if self.config["loss_fn"] == "infoNCE":
                loss = model.infoNCELoss(gene_features, mol_features, logit, device) # logit is inv_tau
            elif self.config["loss_fn"] == "cloome":
                loss = model.cloob(gene_features, mol_features, logit, self.config["hopfield_input_dim"], self.config["hopfield_scale"], device)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), gene_features.size(0))

            # clamp inv_tau?
            if self.config["inv_tau_clamp"]:
                model.logit_inv_tau.data.clamp_(min=0.1, max=self.config["inv_tau_max"])

        # return train_loss.avg, logit
        return train_loss.avg, logit
    
    def val_epoch(self,model, dataloader, device, scheduler=None):
    
        model.eval() # turn off dropout
        val_loss = AvgMeter()

        for batch_idx, batch in enumerate(dataloader):
                
                image = batch["image"].to(device) # gene expression
                mol_embed = batch["smiles_emb"].to(device)
                smiles = batch["smiles"]
                dosage = batch["dosage"].to(device)

                if(self.config['add_dosage']):
                    gene_features, mol_features, logit = model(image, mol_embed, dosage) # it returns unnormed embeddings
                else:
                    gene_features, mol_features, logit = model(image, mol_embed, None)

                # compute loss
                if self.config["loss_fn"] == "infoNCE":
                    loss = model.infoNCELoss(gene_features, mol_features, logit,  device) # logit is inv_tau
                elif self.config["loss_fn"] == "cloome":
                    loss = model.cloob(gene_features, mol_features, logit, self.config["hopfield_input_dim"], self.config["hopfield_scale"], device)

                val_loss.update(loss.item(), gene_features.size(0))
        
        # update learning rate
        if scheduler:
            scheduler.step(val_loss.avg)

        return val_loss.avg, logit
    
    def _train(self, model, train_loader, val_loader):

        DEVICE = next(model.parameters()).device
        USE_WANDB = self.config['use_wandb']
        if USE_WANDB:
            wandb.init(project=self.config['wandb_project'], 
                       name=f"{self.config['experiment_ID']}",
                       entity=self.config['wandb_entity'])
            wandb.config.update(self.config)

        num_epochs = self.config['num_epochs']
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               mode='min', 
                                                               factor=0.1, patience=7, verbose=True)

        print('starting training...')

        val_checkpoint = os.path.join(self.config['checkpoint_dir'], self.config['model_chkpt_dir'], self.config['experiment_ID'])
        patience, delta = self.config['patience'], self.config['delta']
        early_stopping = EarlyStopping(save_path=val_checkpoint, save_frequency= self.config["save_every"], patience=patience, delta=delta)
        
        for epoch in range(num_epochs):

            # train epoch loss
            train_loss, train_logit = self.train_epoch(model, train_loader, DEVICE, optimizer, scheduler)
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss}")

            if USE_WANDB:
                wandb.log({"train_loss": train_loss}, step = epoch)
                wandb.log({"inv_tau": train_logit}, step=epoch)
            
            # val epoch loss
            val_loss, logit = self.val_epoch(model, val_loader, DEVICE)
            print(f"Epoch: {epoch+1}, Val Loss: {val_loss}")

            if USE_WANDB:
                wandb.log({"val_loss": val_loss}, step = epoch)
            
            # early_stopping to save model checkpoint
            early_stopping(val_loss, train_loss, model, epoch, optimizer)

            if early_stopping.early_stop: 
                print("Early stopping")
                break


    def fit(self, model, train_loader, val_loader):
        self._train(model, train_loader, val_loader)
        return model
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
