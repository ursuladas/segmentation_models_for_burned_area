import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import segmentation_models_pytorch as smp
import glob
import xarray as xr
import pickle
# Import models from models.py
from seg_models import UnetModel, FPNModel, PSPNetModel, DeepLabV3Model, PANModel
import wandb
import matplotlib.pyplot as plt

# Import your dataset class
from datamodule import SegmentationDataset  # Replace with your actual dataset class

# Training function
def train(model, loader, optimizer, criterion,device,epoch):
    model.train()
    train_losses=[]
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(loader)):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        # print(f'x_batch {x_batch.shape} and y_batch {y_batch.shape}')
        # Forward pass
        predictions = model(x_batch)
        # print(f'prediction {predictions.shape}')
        loss = criterion(predictions, y_batch)
        predictions=torch.nn.functional.sigmoid(predictions)
        threshold = 0.3
        predictions = (predictions > threshold).float()

        if batch_idx == len(loader)-1:
            log_predictions(predictions[:, 0], y_batch[:, 0], epoch, prefix='train')

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        wandb.log({"epoch":epoch,"Train Step Loss":loss.item()})

    avg_loss = sum(train_losses) / len(train_losses)
    return predictions,avg_loss

# Evaluation function
def evaluate(model, loader, criterion,device,epoch):
    model.eval()
    val_losses=[]
    with torch.no_grad():
        for batch_idx,(x_val, y_val) in enumerate(tqdm(loader)):
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            # Forward pass
            predictions = model(x_val)
            loss = criterion(predictions, y_val)
            predictions=torch.nn.functional.sigmoid(predictions)
            threshold = 0.3
            predictions = (predictions > threshold).float()
            val_losses.append(loss.item())
            if batch_idx == len(loader)-1:
                log_predictions(predictions[:, 0], y_val[:, 0], epoch, prefix='val')
            wandb.log({"epoch":epoch,"Val Step Loss":loss.item()})

    avg_loss = sum(val_losses) / len(val_losses)

    return predictions,avg_loss


def log_predictions(predictions, ground_truths, step, prefix='train', num_samples=2):
    # Randomly select `num_samples` indices from the batch
    batch_size = predictions.shape[0]
    random_indices = torch.randint(0, batch_size, (num_samples,))

    for i, idx in enumerate(random_indices):
        fig, ax = plt.subplots(1, 2, figsize=(20, 12))

        # Plot ground truth
        im1 = ax[0].imshow(ground_truths[idx].cpu().numpy(), cmap='gray')
        ax[0].set_title(f'{prefix.capitalize()} Ground Truth {i+1}')
        fig.colorbar(im1, ax=ax[0])  # Attach colorbar to the ground truth plot

        # Plot prediction
        im2 = ax[1].imshow(predictions[idx].cpu().numpy(), cmap='gray')
        ax[1].set_title(f'{prefix.capitalize()} Prediction {i+1}')
        fig.colorbar(im2, ax=ax[1])  # Attach colorbar to the prediction plot

        # Log the figure to wandb
        wandb.log({f"{prefix}_sample_{i+1}_epoch_{step}": wandb.Image(fig)})
        plt.close(fig)

def main(args):

    wandb.init(project='Segmentation',name=args.desc,config=args)
    config=wandb.config

    train_paths=[]
    val_paths=[]
    for year in args.train_years:
        train_paths.extend(glob.glob(f'{args.root_dir}/{year}/*.nc'))

    for year in args.val_years:
        val_paths.extend(glob.glob(f'{args.root_dir}/{year}/*.nc'))
        
    train_data= [xr.open_dataset(fp) for fp in train_paths]
    val_data= [xr.open_dataset(fp) for fp in val_paths]
    
    # print(f' Validation Data Length {len(val_data)}, \n Train Data Length {len(train_data)} ')
    
    with open(args.train_stat_dict, 'rb') as f:
            train_stat = pickle.load(f)

    with open(args.val_stat_dict, 'rb') as f:
            val_stat = pickle.load(f)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device='cpu'

    # Load your dataset
    train_dataset = SegmentationDataset(train_data,train_stat,args.target,args.input_vars,crop_size=args.crop_size,stat='min_max')
    val_dataset = SegmentationDataset(val_data,val_stat,args.target,args.input_vars,crop_size=args.crop_size,stat='min_max')

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False)

    # Instantiate the model
    model_cls = {"Unet": UnetModel, "FPN": FPNModel, "PSPNet": PSPNetModel, "DeepLabV3": DeepLabV3Model, "PAN": PANModel}
    model = model_cls[args.model_name](encoder_name=args.encoder_name, in_channels=len(args.input_vars), classes=1)
    print(f'Model is {model}')
    model=model.to(device)

    # Loss and optimizer
    loss=args.loss
    if loss=='BCE_raw':
        criterion = nn.BCEWithLogitsLoss()
    elif loss=='BCE':
        criterion=nn.BCELoss()
    elif loss=='dice':
        criterion=smp.losses.DiceLoss(mode='binary', log_loss=False, from_logits=True,eps=1e-07)
    elif loss=='focal':
        criterion=smp.losses.FocalLoss(mode='binary', alpha=None, gamma=2.0)
    elif loss=='ts':
        criterion=smp.losses.TverskyLoss(mode='binary', classes=1, log_loss=False, from_logits=True,eps=1e-07, alpha=0.3, beta=0.7, gamma=1.0)[source]

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train for one epoch
        train_prediction,train_loss = train(model, train_loader, optimizer, criterion,device,epoch)
        print(f"Training loss: {train_loss:.4f}")

        # Validate
        val_prediction,val_loss = evaluate(model, val_loader, criterion,device,epoch)
        print(f"Validation loss: {val_loss:.4f}")

        # Log losses to TensorBoard
        # writer.add_scalar('Loss/Train', train_loss, epoch)
        # writer.add_scalar('Loss/Validation', val_loss, epoch)
        # Log metrics to W&B
        wandb.log({"Training Loss": train_loss, "Validation Loss": val_loss})
        

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_{args.model_name}.pth')
            print("Saved best model")
            wandb.save(f'best_model_{args.model_name}.pth')

        
    wandb.finish()
    # writer.close()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Training script for segmentation models')

    # Hyperparameters
    parser.add_argument('--root_dir',type=str,default='/home/udas/Desktop/UD_Data_Copy/b.final_burned_area',help='directory containing all the samples')
    parser.add_argument('--train_stat_dict',type=str,default='/home/udas/Desktop/UD_Data_Copy/Spatial_Models/train_stats.pkl',help='minmax values of train, rerun if using different years')
    parser.add_argument('--val_stat_dict',type=str,default='/home/udas/Desktop/UD_Data_Copy/Spatial_Models/val_stats.pkl',help='minmax values of val, rerun script if using different years')

    parser.add_argument('--input_vars',type=str,nargs='+',default=['ignition_points','ssrd','smi','d2m','t2m','wind_speed'],help='input features')
    parser.add_argument('--target',type=str,default='burned_areas',help='target feature')
   
    parser.add_argument('--crop_size',type=int,default=32,help='crop size')
    parser.add_argument('--desc',type=str,default='train',help='run description')
    parser.add_argument('--train_years',type=int,nargs='+',default=[2017,2018,2019,2020],help='train_years')
    parser.add_argument('--val_years',type=int,nargs='+',default=[2022],help='train_years')

    
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--loss', type=str, default='BCE_raw', help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--encoder_name', type=str, default='resnet34', help='Name of the encoder')
    parser.add_argument('--model_name', type=str, default='FPN', choices=['Unet', 'FPN', 'PSPNet', 'DeepLabV3', 'PAN'], help='Segmentation model to use')


    # Parse arguments
    args = parser.parse_args()
    print(f'input vars {args.input_vars} and {type(args.input_vars)}')
    print(f'train years {args.target} and {type(args.target)}')
    print(f'val years {args.val_years} and {type(args.val_years)}')
    print(f'train years {args.train_years} and {type(args.train_years)}')

    main(args)

