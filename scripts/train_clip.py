import os
import argparse
import yaml

from models.clip.model import CLIP, projection, ImageEncoder
from models.clip.trainer import CLIPTrainer
from utils.data_loader import CLIPDataLoader
from utils.data_utils import CustomTransform

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def str_to_list(x):
    return list(map(int, x.split(',')))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP")

    # data args
    data_args = parser.add_argument_group("Data Args")  

    data_args.add_argument("--data_dir", type=str, default="/rohlan/workspace/data/bbbc021_all") 
    data_args.add_argument("--meta_data_csv", type=str, default="bbbc021_df_all.csv")
    data_args.add_argument("--embeddings_file", type=str, default="unique_smiles_morgan_fingerprints.pkl")
    data_args.add_argument("--transform_obj", type=str, default="CustomTransform")
    data_args.add_argument('--checkpoint_dir', type=str, default='/rohlan/workspace/checkpoints/')
    data_args.add_argument('--experiment_ID', type=str, required=True) 
    data_args.add_argument("--model", type=str, default='clip') # -- checkpoints/<model>

    # train args 
    train_args = parser.add_argument_group("Train Args")

    train_args.add_argument('--wandb_project', type=str, default='ai_project')
    train_args.add_argument("--wandb_entity", type=str, default="res-ba-org")
    train_args.add_argument("--use_wandb", type=str2bool, default='False')
    train_args.add_argument("--train_batch_size", type=int, default=256)
    train_args.add_argument("--val_batch_size", type=int, default=256)
    train_args.add_argument("--num_epochs", type=int, default=200)
    train_args.add_argument("--lr", type=float, default=0.0001)
    train_args.add_argument("--patience", type=int, default=10)
    train_args.add_argument("--delta", type=float, default=0.0001)
    train_args.add_argument('--device_id', type=str, default='1')
    train_args.add_argument("--device_ids", type=str_to_list, default="1, 2, 3, 4")
    train_args.add_argument("--distributed", type=str2bool, default="False")
    train_args.add_argument("--save_every", type=int, default=0)    
    
    # model args

    # image projection head
    image_proj_args = parser.add_argument_group("Image Projection Head Args")

    image_proj_args.add_argument("--image_proj_hidden_sizes", type=str_to_list, default="256,128")
    image_proj_args.add_argument("--image_proj_input_size", type=int, default=256)
    image_proj_args.add_argument("--image_proj_output_size", type=int, default=64)

    # mol projection head
    mol_proj_args = parser.add_argument_group("Mol Projection Head Args")

    mol_proj_args.add_argument("--mol_proj_hidden_sizes", type=str_to_list, default="512, 256,128")
    mol_proj_args.add_argument("--mol_proj_input_size", type=int, default=2048)
    mol_proj_args.add_argument("--mol_proj_output_size", type=int, default=64)

    # clip_model args
    clip_args = parser.add_argument_group("CLIP Model Args")

    clip_args.add_argument("--freeze_encoder", type=str2bool, default="False")
    clip_args.add_argument("--ge_type", type=str, default="median")
    clip_args.add_argument("--learnable_inv_tau", type=str2bool, default="True")
    clip_args.add_argument("--inv_tau", type=float, default=14.3)
    clip_args.add_argument("--loss_fn", type=str, default="infoNCE", choices=["infoNCE", "cloome"])
    clip_args.add_argument("--hopfield_scale", type=float, default=0.5)
    clip_args.add_argument("--hopfield_input_dim", type=int, default=64)
    clip_args.add_argument("--inv_tau_clamp", type=str2bool, default="False")
    clip_args.add_argument("--inv_tau_max", type=float, default=4.6052)
    clip_args.add_argument("--use_attention", type=str2bool, default="False")

    
    


    # config = parser.parse_args(args=["--experiment_ID","1"]) 
    # assert config, 'comment debug config'
    config = parser.parse_args()
    return vars(config)


def main(config_file):
    
    # save configuration
    config_save_path = os.path.join(config_file['checkpoint_dir'], config_file['model'], f"{config_file['experiment_ID']}")
    os.makedirs(config_save_path, exist_ok=True)

    # save config
    with open(f"{config_save_path}/config.yaml", "w") as f:
        yaml.dump(config_file, f)

    print(f"experiment: {config_file['experiment_ID']} config file saved at {config_save_path}")
    
    # load data
    if config_file['transform_obj'] == "CustomTransform":
        transform = CustomTransform()
    # device
    device = torch.device(f"cuda:{config_file['device_id']}" if torch.cuda.is_available() else "cpu")

    print("creating dataloaders ...")
    data_loader = CLIPDataLoader(device, transform, **config_file)
    train_loader, val_loader = data_loader.train_dataloader(), data_loader.test_dataloader()
    print("data loaded!")
    
    # model
    # gex_vae_config_file
    print("loading models ...")
    image_encoder = ImageEncoder().to(device)

    # image projection head
    image_projection = projection(
        input_dim=config_file['image_proj_input_size'],
        hidden_sizes=config_file['image_proj_hidden_sizes'],
        output_dim=config_file['image_proj_output_size']
    ).to(device)
    print("image projection head with hidden sizes: ", config_file['image_proj_hidden_sizes'], "loaded!")
    # smiles projection network
    mol_projection = projection(
        input_dim=config_file['mol_proj_input_size'],
        hidden_sizes=config_file['mol_proj_hidden_sizes'],
        output_dim=config_file['mol_proj_output_size']
    ).to(device)
    print("mol projection head with hidden sizes: ", config_file['mol_proj_hidden_sizes'], "loaded!")

    # final clip model
    model = CLIP(
        image_encoder=image_encoder,
        init_inv_tau=config_file['inv_tau'],
        learnable_inv_tau=config_file['learnable_inv_tau'],
        image_projection_network=image_projection,
        mol_projection_network=mol_projection
    ).to(device)

    print("CLIP model loaded!")

    if config_file['freeze_encoder']:
        model.freeze_encoder()
        print("image encoder freezed!")
    else:
        model.unfreeze_encoder()
        print("image encoder unfreezed!")

    model.device = device

    # trainer
    trainer = CLIPTrainer(config_file)

    trainer.fit(model, train_loader, val_loader)



if __name__ == "__main__":

    config = parse_args()
    main(config)