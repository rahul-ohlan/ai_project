import os
import numpy as np
import argparse, yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP


# SimCLR
from models.simCLR import simCLR, ImageEncoder, simCLRTrainer, simCLRrn50
from models.simCLR.modules import NT_Xent
from models.simCLR.modules.transformations import TransformsSimCLR
from models.simCLR.modules.sync_batchnorm import convert_model
from utils.data_loader import simCLRDataLoader #  -- to be implemented
from utils.model_utils import load_optimizer




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
    parser = argparse.ArgumentParser(description="Train simCLR")

    # data args
    data_args = parser.add_argument_group("Data Args")  
    data_args.add_argument("--data_dir", type=str, default="/rohlan/workspace/data/bbbc021_all") 
    data_args.add_argument("--meta_data_csv", type=str, default="bbbc021_df_all.csv")
    data_args.add_argument("--transform_obj", type=str, default="CustomTransform")
    data_args.add_argument('--checkpoint_dir', type=str, default='/rohlan/workspace/checkpoints/')
    data_args.add_argument('--experiment_ID', type=str, required=True) 
    data_args.add_argument("--model_chkpt_dir", type=str, default='simCLR') # -- checkpoints/<model>

    # train args 
    train_args = parser.add_argument_group("Train Args")

    train_args.add_argument('--wandb_project', type=str, default='ai_project_colab')
    train_args.add_argument("--wandb_entity", type=str, default="ai_project_colab")
    train_args.add_argument("--use_wandb", type=str2bool, default='True')
    train_args.add_argument("--batch_size", type=int, default=256)
    train_args.add_argument("--num_epochs", type=int, default=100)
    train_args.add_argument("--lr", type=float, default=0.0005)
    train_args.add_argument("--patience", type=int, default=25)
    train_args.add_argument("--delta", type=float, default=0.0001)
    train_args.add_argument('--device_id', type=str, default='1')
    train_args.add_argument("--gpu_ids", type=str_to_list, default="0, 1, 5, 6")
    train_args.add_argument("--nodes", type=int, default=1)
    train_args.add_argument("--distributed", type=str2bool, default="True")
    train_args.add_argument("--save_every", type=int, default=0)
    train_args.add_argument("--weight_decay", type=float, default=1e-5)  
    train_args.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "LARS"])  
    train_args.add_argument("--use_scheduler", type=str2bool, default="False")
    
    # model args

    # image projection head
    image_proj_args = parser.add_argument_group("Image Projection Head Args")

    image_proj_args.add_argument("--n_features", type=int, default=256) # -- output of resnet encoder from IMPA
    image_proj_args.add_argument("--projection_dim", type=int, default=256) # final projection for contrastive learning
    image_proj_args.add_argument("--image_size", type=int, default=96) # size of images
    image_proj_args.add_argument("--resnet_model", type=str, default="resnet50", choices=["resnetIMPA", "resnet50"]) # resnet model to use


    # simCLR model args
    simCLR_args = parser.add_argument_group("simCLR Model Args")

    simCLR_args.add_argument("--temperature", type=float, default=0.5)


    
    


    # config = parser.parse_args(args=["--experiment_ID","1"]) 
    # assert config, 'comment debug config line above'
    config = parser.parse_args()
    return vars(config)


def main(rank, config_file):
    
    if config_file['distributed']:
        dist.init_process_group(backend='nccl', rank=rank, world_size=config_file['world_size'])
    
    # save configuration
    config_save_path = os.path.join(config_file['checkpoint_dir'], config_file['model_chkpt_dir'], f"{config_file['experiment_ID']}")
    os.makedirs(config_save_path, exist_ok=True)

    # save config
    with open(f"{config_save_path}/config.yaml", "w") as f:
        yaml.dump(config_file, f)

    print(f"experiment: {config_file['experiment_ID']} config file saved at {config_save_path}")
    
    # load data
    if config_file['transform_obj'] == "CustomTransform":
        transform = TransformsSimCLR(size=config_file['image_size'])

        # device
    # device = torch.device(f"cuda:{config_file['device_id']}" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")


    print("creating dataloaders ...")
    data_loader = simCLRDataLoader(rank, transform, **config_file) # samples will need rank since will use DDP and distributed data sampler
    train_loader = data_loader.train_dataloader()
    print("data loaded!")
    
    # model
    # gex_vae_config_file
    print("loading models ...")
    image_encoder = ImageEncoder().to(device)

    if config_file['resnet_model'] == "resnetIMPA":
        model = simCLR(encoder=image_encoder,
                    projection_dim=config_file['projection_dim'],
                    n_features=config_file['n_features'])
    else:
        model = simCLRrn50(projection_dim=config_file['projection_dim'])


    model = model.to(device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(config_file, model)
    criterion = NT_Xent(config_file['batch_size'], config_file['temperature'], config_file['world_size'])

    # now put wrap model with DDP
    if config_file['distributed']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])

    # trainer

    model.device = device
    trainer = simCLRTrainer(config_file, rank)

    trainer.fit(model, train_loader, optimizer, criterion, scheduler)



if __name__ == "__main__":

    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,args['gpu_ids']))
    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    # if not os.path.exists(args['model_path']):
    #     os.makedirs(args.model_path)

    args['num_gpus'] = len(args['gpu_ids'])
    args['world_size'] = args['num_gpus'] * args['nodes']

    
    mp.spawn(main, args=(args,), nprocs=args['num_gpus'], join=True)
    # else:
    #     main(0, args)