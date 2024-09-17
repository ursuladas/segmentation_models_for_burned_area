import cv2 as cv
import numpy as np
import xarray as xr
from datamodule import SegmentationDataset

def find_area(mask):
    resolution=1
    num_fire_cells=np.count_nonzero(mask)
    return (num_fire_cells * resolution)

def find_centroids(mask):
    # Find connected components
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    
    # Exclude the background (label 0)
    centroids = centroids[1:]  # Remove the centroid of the background
    return centroids

def visualise_centroid(mask, centroids):
    # Create an empty color image
    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Map binary values to colors
    mask_color[mask == 1] = [0, 0, 255]  # Red for 1's
    mask_color[mask == 0] = [0, 100, 0]  # Dark green for 0's

    if centroids is not None:
        for centroid in centroids:
            centroid_x, centroid_y = map(int, centroid)

            # Calculate the size of the cluster (number of pixels with value 1)
            cluster_size = np.sum(mask == 1)
        
            # Choose a smaller radius if the mask has only one pixel
            radius = 0 if cluster_size == 1 else 5
            # Draw a circle at the centroid in black
            cv.circle(mask_color, (centroid_x, centroid_y), 0, (0, 0, 0), -1)  # Black circle

    return mask_color
def compute_area_mse(predictions,ground_truth):
    area_diff=[]
    for i in range(predictions.shape[0]):
        pred_mask = predictions[i, 0, :, :].detach().to('cpu').numpy()
        gt_mask = ground_truth[i, 0, :, :].detach().to('cpu').numpy()

        # Find centroids for both prediction and ground truth masks
        pred_centroid = find_area(pred_mask)
        gt_centroid = find_area(gt_mask)

        area_mse=np.mean(np.square(pred_centroid-gt_centroid))
        area_diff.append(area_mse)

    return area_diff


def compute_centroid_mse(predictions, ground_truth):
    centroid_distances = []

    for i in range(predictions.shape[0]):
        # Get the prediction and ground truth masks for the i-th sample in the batch
        pred_mask = predictions[i, 0, :, :].detach().to('cpu').numpy()
        gt_mask = ground_truth[i, 0, :, :].detach().to('cpu').numpy()

        # Find centroids for both prediction and ground truth masks
        pred_centroid = find_centroids(pred_mask)
        gt_centroid = find_centroids(gt_mask)

        pred_centroid_vis=visualise_centroid(pred_mask,pred_centroid)
        cv.imwrite('pred_centroid.png',pred_centroid_vis)

        gt_centroid_vis=visualise_centroid(gt_mask,gt_centroid)
        cv.imwrite('gt_centroid.png',gt_centroid_vis)

        # Check if both centroids exist
        if pred_centroid is not None and gt_centroid is not None:
            # Calculate the Euclidean distance between centroids
            if pred_centroid.shape[0]== gt_centroid.shape[0]:
                distance = np.linalg.norm(pred_centroid - gt_centroid)
                centroid_distances.append(distance)

    # Calculate Mean Squared Error (MSE) for centroid distances
    if len(centroid_distances) > 0:
        centroid_mse = np.mean(np.square(centroid_distances))
    else:
        centroid_mse = None  # Handle case with no valid centroids

    return centroid_mse

def main(mask):
    centroids=find_centroids(mask)
    print(centroids)
    mask_centroid=visualise_centroid(mask,centroids)
    cv.imwrite('centroid.png',mask_centroid)
    # cv.waitKey(2)
    # cv.destroyAllWindows()


if __name__=="__main__":

    ds=xr.open_dataset('/home/udas/Desktop/UD_Data_Copy/b.final_burned_area/2008/sample_1622.nc')
    ia=ds['ignition_points'].isel(time=-1).where(ds['ignition_points'] == 0, 1)
    ia=ia.values.squeeze()
      # Find coordinates where ia is non-zero
    non_zero_coords = np.nonzero(ia)

    # Zip the results to get list of (row, column) coordinates
    coordinates = list(zip(non_zero_coords[0], non_zero_coords[1]))
    print(coordinates)
    cv.imwrite('ia.png',ia*255)
    da=ds['burned_areas'].isel(time=-1).where(ds['burned_areas'] == 0, 1)
    da=da.values.squeeze()
    cv.imwrite('da.png',da*255)
    print(f'{type(da)} andf {da.shape}')
    main(da)