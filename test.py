import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np

from torchmetrics import F1Score,Dice
from torchmetrics.segmentation import HausdorffDistance, MeanIoU
from torchmetrics.classification import BinaryAccuracy,BinaryAUROC,BinaryCohenKappa,BinaryAveragePrecision,BinaryF1Score
from torchmetrics.regression import MeanSquaredError
import torch.nn.functional as F
from tqdm import tqdm
import segmentation_models_pytorch as smp
import glob
import xarray as xr
import pickle
import matplotlib.pyplot as plt
# Import models from models.py
from seg_models import UnetModel, FPNModel, PSPNetModel, DeepLabV3Model, PANModel,UnetPlusPlusModel,MAnetModel,DeepLabV3PlusModel
import cv2
from qual_metrics import compute_centroid_mse,compute_area_mse
import random

# Import your dataset class
from datamodule import SegmentationDataset  # Replace with your actual dataset class

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Evaluation function
def evaluate(wandb_flag,model, loader, criterion,device,loss_type,threshold):
    model.eval()
    test_losses=[]
    
    centroid_mse_list=[]
    vis_count=[]
    
    # Initialize metrics
    auroc = BinaryAUROC().to(device)
    auprc = BinaryAveragePrecision().to(device)
    f1 = BinaryF1Score(threshold=threshold,).to(device)
    accuracy = BinaryAccuracy(threshold=threshold,multidim_average='global').to(device)
    dice = Dice(threshold=threshold).to(device)
    bin_kappa=BinaryCohenKappa(threshold=threshold).to(device)
    # hausdorff_distance = HausdorffDistance(distance_metric="euclidean",num_classes=2)
    # miou = MeanIoU(num_classes=2,input_format='index').to(device)  # For binary IoU
    mse=MeanSquaredError().to(device)

    # auroc = AUROC(task="binary").to(device)
    # auprc = AveragePrecision(task="binary").to(device)
    # f1 = F1Score(task="binary").to(device)
    # accuracy = Accuracy(task="binary").to(device)
    # miou = MeanIoU(num_classes=2).to(device)  # For binary IoU
    # mse=MeanSquaredError().to(device)


    with torch.no_grad():
        for batch_idx,(x_val, y_val) in enumerate(loader):
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            test_batch=x_val.shape[0]
            # Forward pass
            raw_predictions = model(x_val)
            predictions=torch.nn.functional.sigmoid(raw_predictions)

            if loss_type in ['BCE_raw','dice_raw','ts_raw','focal','weigh_bce_raw','jaccard_raw']:
                loss = criterion(raw_predictions, y_val)
            else:
                loss = criterion(predictions, y_val)

            test_losses.append(loss.item())
            mse.update(predictions,y_val)

            

            # Update metrics
            auroc.update(predictions, y_val.int())
            auprc.update(predictions, y_val.int())
            f1.update(predictions, y_val.int())
            dice.update(predictions, y_val.int())
            bin_kappa.update(predictions, y_val.int())

            accuracy.update(predictions, y_val.int())
            thresh_predictions = (predictions > threshold).float()
            
            # print(f'batch size {test_batch} prediction shape {predictions.shape} and target shape {y_val.int().shape}')
            # exit(0)
            if test_batch==1 and vis_count<=50:
                log_predictions(wandb_flag,predictions,  y_val.int(), batch_idx)
                vis_count+=1
            elif test_batch>1 and batch_idx == len(loader)-1:
                log_predictions(wandb_flag,predictions,  y_val.int(), batch_idx)
            
            if wandb_flag:
                wandb.log({"batch":batch_idx,"Test Step Loss":loss.item()})
                
            # Compute centroid MSE for the batch
            # miou.update(predictions.int(), y_val.int())
    

            centroid_mse = compute_centroid_mse(thresh_predictions, y_val)
            area_mse=compute_area_mse(thresh_predictions,y_val)

            if centroid_mse is not None:
                centroid_mse_list.append(centroid_mse)
            # print(f'{predictions} {y_val} {miou.compute()}')

    avg_loss = sum(test_losses) / len(test_losses)
    avg_centroid_mse = sum(centroid_mse_list) / len(centroid_mse_list) if len(centroid_mse_list) > 0 else None
    avg_area_mse = sum(area_mse) / len(area_mse)

  # Compute the final values of the metrics
    final_auroc = auroc.compute()
    final_auprc = auprc.compute()
    final_f1 = f1.compute()
    final_accuracy = accuracy.compute()
    final_dice = dice.compute()
    final_mse=mse.compute()
    final_bin_kappa=bin_kappa.compute()
    # final_iou = miou.compute()
    # final_hdist=hausdorff_distance.compute()
    final_hdist=0
    # final_iou = miou.compute()
    final_iou=0
    # Reset metrics to clear for the next evaluation
    auroc.reset()
    auprc.reset()
    f1.reset()
    accuracy.reset()
    dice.reset()
    mse.reset()
    bin_kappa.reset()
    # miou.reset()

    metrics={"area_mse": avg_area_mse, "centroid_distance_mse": avg_centroid_mse,"miou":final_iou,"hausdorff_distance":final_hdist,"binary kappa":final_bin_kappa,"mse":final_mse,"auprc":final_auprc,"auroc":final_auroc,"f1":final_f1,"dice":final_dice,"accuracy":final_accuracy}
    
    return predictions,metrics

def log_predictions(wandb_flag,predictions, ground_truths, step, prefix='test', num_samples=50):
    # Define the grid size (32x32) for each graph
    grid_size = (32, 32)
    
    if predictions.shape[0]>1:
          # Randomly select `num_samples` indices from the batch
            # batch_size = predictions.shape[0] // 1024  # Each graph has 1024 nodes
            random_indices = torch.randint(0, predictions.shape[0], (num_samples,))
    else:
        # In this case, we have only one graph, so no need to randomize
        random_indices = [0]  # Only one possible index (0)
  

    for i, idx in enumerate(random_indices):
    # for i in range(batch_size):
        fig, ax = plt.subplots(1, 2, figsize=(20, 12))


        ground_truth_grid = ground_truths[idx,0,:,:].cpu().numpy()
        prediction_grid = predictions[idx,0,:,:].detach().cpu().numpy()
        # print(f'ground_truth_grid {ground_truth_grid} prediction shape {prediction_grid.shape}')


        # Plot ground truth
        im1 = ax[0].imshow(ground_truth_grid, cmap='summer')
        ax[0].set_title(f'{prefix.capitalize()} Ground Truth {i+1}')
        fig.colorbar(im1, ax=ax[0])  # Attach colorbar to the ground truth plot

        # Plot prediction
        im2 = ax[1].imshow(prediction_grid, cmap='summer')
        ax[1].set_title(f'{prefix.capitalize()} Predicted{i+1}')
        fig.colorbar(im2, ax=ax[1])
        
        if wandb_flag:
            wandb.log({f"{prefix}_sample_{i+1}": wandb.Image(fig)})
            
        plt.close(fig)
        
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


    SEED = 42  # You can set any integer seed here
    set_seed(SEED)

    #Initiate wandb 
    if args.wandb:
        wandb.init(project='Test_Segmentation',name=args.desc,config=args)
        config=wandb.config

    test_paths=[]
    for year in args.test_years:
        test_paths.extend(glob.glob(f'{args.root_dir}/{year}/*.nc'))
    
    test_data= [xr.open_dataset(fp) for fp in test_paths]
    
    # print(f' Validation Data Length {len(val_data)}, \n Train Data Length {len(train_data)} ')
    
    with open(args.test_stat_dict, 'rb') as f:
            test_stat = pickle.load(f)

    print(f' Print Arguments :{args.input_vars}')

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device='cpu'

    # Load your dataset
    test_dataset = SegmentationDataset(test_data,test_stat,args.target,args.input_vars,crop_size=args.crop_size,stat='min_max')
    
    # Create DataLoader for training and validation
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
    
    print(f'Length of Dataset is {len(test_dataset)}')
    print(f' Length of Loader is {len(test_dataloader)}')
    
    # Instantiate the model
    model_cls = {"Unet": UnetModel, "FPN": FPNModel, "PSPNet": PSPNetModel, "DeepLabV3": DeepLabV3Model, "PAN": PANModel, "Unet++":UnetPlusPlusModel,"MAnet":MAnetModel,"DeepLabV3+":DeepLabV3Model}
    model = model_cls[args.model_name](encoder_name=args.encoder_name, in_channels=len(args.input_vars), classes=1)
    model.load_state_dict(torch.load(args.model_path,weights_only=True))
    model=model.to(device)

    # Loss and optimizer
    loss=args.loss
    
    if loss=='BCE_raw':
        criterion = nn.BCEWithLogitsLoss()
    elif loss=='BCE':
        criterion=nn.BCELoss()
    elif loss=='dice_raw':
        criterion=smp.losses.DiceLoss(mode='binary', log_loss=False, from_logits=True,eps=1e-07)
    elif loss=='dice':
        criterion=smp.losses.DiceLoss(mode='binary', log_loss=False, from_logits=False,eps=1e-07)
    elif loss=='focal':   
        criterion=smp.losses.FocalLoss(mode='binary', alpha=None, gamma=2.0)
    elif loss=='ts_raw':
        criterion=smp.losses.TverskyLoss(mode='binary',log_loss=False, from_logits=True,eps=1e-07, alpha=0.4, beta=0.6, gamma=1.0)
    elif loss=='ts':
        criterion=smp.losses.TverskyLoss(mode='binary',log_loss=False, from_logits=False,eps=1e-07, alpha=0.4, beta=0.6, gamma=1.0)
    elif loss == 'weigh_bce_raw':
        pos_weight = torch.tensor([args.pos_weight]).to(device)  # Assuming `args.pos_weight` is passed for positive weighting
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss == 'weigh_bce':
        pos_weight = torch.tensor([args.pos_weight]).to(device)  # Positive weight for weighted BCE
        criterion = nn.BCELoss(pos_weight=pos_weight)
    elif loss == 'jaccard_raw':
        criterion = smp.losses.JaccardLoss(mode='binary', from_logits=True)
    elif loss == 'jaccard':
        criterion = smp.losses.JaccardLoss(mode='binary', from_logits=False)
    else:
        raise ValueError("Unsupported loss type provided.")


    test_prediction,metrics = evaluate(args.wandb,model,test_dataloader,criterion,device,loss,args.seg_threshold)
    print(f"Test MSE: {metrics['mse']:.4f} \n"
      f"Test Dice Coefficient: {metrics['dice']:.4f} \n"
      f"Test AUPRC: {metrics['auprc']:.4f} \n"
      f"Test AUROC: {metrics['auroc']:.4f} \n"
      f"Test f1: {metrics['f1']:.4f} \n"
      f"Test Accuracy: {metrics['accuracy']:.4f} \n"
      f"Hausdorff_distance: {metrics['hausdorff_distance']:.4f} \n"
      f"miou: {metrics['miou']:.4f} \n"
      f"area_mse: {metrics['area_mse']:.4f} \n"
      f"centroid_mse: {metrics['centroid_distance_mse']:.4f} \n"
      f"binary kappa :{metrics['binary kappa']:.4f}")

    # test_prediction,test_epoch_loss,test_centroid_mse,test_area_mse,final_auprc, final_auroc, final_f1, final_accuracy, final_iou = evaluate(model, test_loader, criterion,device)
    # print(f"Test Loss: {test_epoch_loss:.6f} and Test Centroid {test_centroid_mse} and Test Area MSE {test_area_mse} \n Test AUPRC {final_auprc} \n Test AUROC {final_auroc} \n Test f1 {final_f1} \n Test Accuracy {final_accuracy}, \n Test IoU {final_iou}")
    # Call the visualization function
    # visualize_predictions('./test_predictions.pt', './test_ground_truth.pt', batch_idx=0)

    
 

    # writer.close()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Training script for segmentation models')

    # Hyperparameters
    parser.add_argument('--root_dir',type=str,default='/home/udas/Desktop/UD_Data_Copy/b.final_burned_area',help='directory containing all the samples')
    parser.add_argument('--test_stat_dict',type=str,default='/home/udas/Desktop/UD_Data_Copy/Spatial_Models/test_stats.pkl',help='minmax values of val, rerun script if using different years')
    parser.add_argument('--stat',type=str,default='min_max',help='normalisation')
    parser.add_argument('--seg_threshold',type=float,default=0.5)
    
    parser.add_argument('--input_vars',type=str,nargs='+',default=['ignition_points','ssrd','smi','d2m','t2m','wind_speed'],help='input features')
    parser.add_argument('--target',type=str,default='burned_areas',help='target feature')
    parser.add_argument('--crop_size',type=int,default=32,help='crop size')
    parser.add_argument('--test_years',type=int,nargs='+',default=[2021,2022],help='train_years')
    # parser.add_argument('--val_years',type=list,default=[2022],help='train_years')
    # parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--loss', type=str, default='BCE_raw', help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--pos_weight',type=float,default=18.5) #weight of positive - inverse of num_pos/num_neg

    # parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--encoder_name', type=str, default='resnet34', help='Name of the encoder')
    parser.add_argument('--model_name', type=str, default='FPN', choices=['Unet', 'FPN', 'PSPNet', 'DeepLabV3', 'PAN','MAnet','Unet++','DeepLabV3+'], help='Segmentation model to use')
    parser.add_argument('--model_path',type=str,default='/home/udas/Desktop/UD_Data_Copy/Segmentation_Models/models/best_model_PSPNet_BCE_raw.pth',help='directory containing all the samples')
    
    parser.add_argument('--desc', type=str, default='test', help='Name of the encoder')
    parser.add_argument('--wandb',action="store_true",help='wanb')



    # Parse arguments
    args = parser.parse_args()

    main(args)

