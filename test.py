import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import AUROC, AveragePrecision, F1Score
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification.accuracy import Accuracy
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

# Import your dataset class
from datamodule import SegmentationDataset  # Replace with your actual dataset class

# Evaluation function
def evaluate(model, loader, criterion,device):
    model.eval()
    val_losses=[]
    centroid_mse_list=[]

    # Initialize metrics
    auroc = AUROC(task="binary").to(device)
    auprc = AveragePrecision(task="binary").to(device)
    f1 = F1Score(task="binary").to(device)
    accuracy = Accuracy(task="binary").to(device)
    miou = MeanIoU(num_classes=2).to(device)  # For binary IoU
    mse=MeanSquaredError().to(device)


    with torch.no_grad():
        for x_val, y_val in loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            # Forward pass
            predictions = model(x_val)
            loss = criterion(predictions, y_val)

            mse.update(predictions,y_val)

            predictions=torch.nn.functional.sigmoid(predictions)

            # Update metrics
            auroc.update(predictions, y_val.int())
            auprc.update(predictions, y_val.int())
            f1.update(predictions, y_val.int())
            
            


            threshold = 0.3
            predictions = (predictions > threshold).float()
            val_losses.append(loss.item())

            # Compute centroid MSE for the batch
            miou.update(predictions.int(), y_val.int())
            accuracy.update(predictions, y_val.int())
            centroid_mse = compute_centroid_mse(predictions, y_val)
            area_mse=compute_area_mse(predictions,y_val)
            if centroid_mse is not None:
                centroid_mse_list.append(centroid_mse)
            # print(f'{predictions} {y_val} {miou.compute()}')

    avg_loss = sum(val_losses) / len(val_losses)
    avg_centroid_mse = sum(centroid_mse_list) / len(centroid_mse_list) if len(centroid_mse_list) > 0 else None
    avg_area_mse = sum(area_mse) / len(area_mse)

  # Compute the final values of the metrics
    final_auroc = auroc.compute()
    final_auprc = auprc.compute()
    final_f1 = f1.compute()
    final_accuracy = accuracy.compute()
    final_iou = miou.compute()

    # Reset metrics to clear for the next evaluation
    auroc.reset()
    auprc.reset()
    f1.reset()
    accuracy.reset()
    miou.reset()

    
    return predictions, avg_loss, avg_centroid_mse, avg_area_mse, final_auprc, final_auroc, final_f1, final_accuracy, final_iou

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
    model_cls = {"Unet": UnetModel, "FPN": FPNModel, "PSPNet": PSPNetModel, "DeepLabV3": DeepLabV3Model, "PAN": PANModel, "Unet++":UnetPlusPlusModel,"MAnet":MAnetModel,"DeepLabV3+":DeepLabV3Model}
    model = model_cls[args.model_name](encoder_name=args.encoder_name, in_channels=len(args.input_vars), classes=1)
    model.load_state_dict(torch.load('/home/udas/Desktop/UD_Data_Copy/Segmentation_Models/models/best_model_Unet++.pth',weights_only=True))
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

    test_prediction,test_epoch_loss,test_centroid_mse,test_area_mse,final_auprc, final_auroc, final_f1, final_accuracy, final_iou = evaluate(model, test_loader, criterion,device)
    print(f"Test Loss: {test_epoch_loss:.6f} and Test Centroid {test_centroid_mse} and Test Area MSE {test_area_mse} \n Test AUPRC {final_auprc} \n Test AUROC {final_auroc} \n Test f1 {final_f1} \n Test Accuracy {final_accuracy}, \n Test IoU {final_iou}")
    # Call the visualization function
    # visualize_predictions('./test_predictions.pt', './test_ground_truth.pt', batch_idx=0)

    
 

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
    parser.add_argument('--model_name', type=str, default='FPN', choices=['Unet', 'FPN', 'PSPNet', 'DeepLabV3', 'PAN','MAnet','Unet++','DeepLabV3+'], help='Segmentation model to use')


    # Parse arguments
    args = parser.parse_args()

    main(args)

