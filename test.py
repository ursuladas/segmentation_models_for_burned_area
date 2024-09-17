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
import matplotlib.pyplot as plt
# Import models from models.py
from seg_models import UnetModel, FPNModel, PSPNetModel, DeepLabV3Model, PANModel

# Import your dataset class
from datamodule import SegmentationDataset  # Replace with your actual dataset class

# Evaluation function
def evaluate(model, loader, criterion,device):
    model.eval()
    val_losses=[]
    with torch.no_grad():
        for x_val, y_val in loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            # Forward pass
            predictions = model(x_val)
            loss = criterion(predictions, y_val)
            predictions=torch.nn.functional.sigmoid(predictions)
            threshold = 0.3
            predictions = (predictions > threshold).float()
            val_losses.append(loss.item())

    avg_loss = sum(val_losses) / len(val_losses)
       # Save the predictions and ground truths
    torch.save(predictions, 'test_predictions.pt')
    torch.save(y_val, 'test_ground_truth.pt')
    
    return predictions,avg_loss

def visualize_predictions(pred_path, gt_path, batch_idx=0):

    # Load saved tensors
    predictions = torch.load(pred_path)
    ground_truth = torch.load(gt_path)

    # Convert to numpy arrays for visualization
    predictions_np = predictions[batch_idx, 0,:,:].detach().to('cpu').numpy()
    ground_truth_np = ground_truth[batch_idx,0,:,:].detach().to('cpu').numpy()

    # Create and save composite image
    create_composite_image(ground_truth_np, predictions_np, colorbar_range=(0, 1))

def create_composite_image(gt_img, pred_img, colorbar_range=None):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display Ground Truth
    cax1 = ax[0].imshow(gt_img, cmap='viridis', vmin=colorbar_range[0], vmax=colorbar_range[1] if colorbar_range else None)
    ax[0].set_title('Ground Truth')
    ax[0].axis('off')

    # Display Prediction
    cax2 = ax[1].imshow(pred_img, cmap='viridis', vmin=colorbar_range[0], vmax=colorbar_range[1] if colorbar_range else None)
    ax[1].set_title('Prediction')
    ax[1].axis('off')

    # Display Colorbar
    cbar = fig.colorbar(cax1, ax=ax[2], orientation='vertical')
    ax[2].set_title('Colorbar')
    ax[2].axis('off')

    plt.tight_layout()
    fig.savefig('pred.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return 

def main(args):

    test_paths=[]
    for year in args.test_years:
        test_paths.extend(glob.glob(f'{args.root_dir}/{year}/*.nc'))
    
    test_data= [xr.open_dataset(fp) for fp in test_paths]
    
    # print(f' Validation Data Length {len(val_data)}, \n Train Data Length {len(train_data)} ')
    
    with open(args.test_stat_dict, 'rb') as f:
            test_stat = pickle.load(f)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device='cpu'

    # Load your dataset
    test_dataset = SegmentationDataset(test_data,test_stat,args.target,args.input_vars,crop_size=args.crop_size,stat='min_max')
    
    # Create DataLoader for training and validation
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
    
    # Instantiate the model
    model_cls = {"Unet": UnetModel, "FPN": FPNModel, "PSPNet": PSPNetModel, "DeepLabV3": DeepLabV3Model, "PAN": PANModel}
    model = model_cls[args.model_name](encoder_name=args.encoder_name, in_channels=len(args.input_vars), classes=1)
    model.load_state_dict(torch.load('/home/udas/Desktop/UD_Data_Copy/Segmentation_Models/best_model.pth'))
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

    test_prediction,test_epoch_loss = evaluate(model, test_loader, criterion,device)
    print("Test Loss: {:.6f}".format(test_epoch_loss))
    # Call the visualization function
    visualize_predictions('./test_predictions.pt', './test_ground_truth.pt', batch_idx=0)

    
 

    # writer.close()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Training script for segmentation models')

    # Hyperparameters
    parser.add_argument('--root_dir',type=str,default='/home/udas/Desktop/UD_Data_Copy/b.final_burned_area',help='directory containing all the samples')
    parser.add_argument('--test_stat_dict',type=str,default='/home/udas/Desktop/UD_Data_Copy/Spatial_Models/val_stats.pkl',help='minmax values of val, rerun script if using different years')
    parser.add_argument('--input_vars',type=list,default=['ignition_points','ssrd','smi','d2m','t2m','wind_speed'],help='input features')
    parser.add_argument('--target',type=str,default='burned_areas',help='target feature')
    parser.add_argument('--crop_size',type=int,default=32,help='crop size')
    parser.add_argument('--test_years',type=list,default=[2016],help='train_years')
    # parser.add_argument('--val_years',type=list,default=[2022],help='train_years')
    # parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--loss', type=str, default='BCE_raw', help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    # parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--encoder_name', type=str, default='resnet34', help='Name of the encoder')
    parser.add_argument('--model_name', type=str, default='Unet', choices=['Unet', 'FPN', 'PSPNet', 'DeepLabV3', 'PAN'], help='Segmentation model to use')


    # Parse arguments
    args = parser.parse_args()

    main(args)

