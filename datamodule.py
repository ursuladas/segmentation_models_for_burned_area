from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import torch
import random


class SegmentationDataset(Dataset):
    def __init__(self, datasets,stat_dict,target,input_vars,crop_size=0,rotate=False,stat='mean_std'):
        """
        Args:
            datasets (list of xarray datasets): List of monotemporal datasets 
            sample_list = list of all netcdf files in train/val/test dataset
            min_max_dict = min max values of all features
            target = define target variable
            input_vars = features of interest
            crop_size = gridsize to pass to model
        """
        self.sample_list = datasets
        self.stat_dict = stat_dict
        self.stat=stat
        self.input_vars=input_vars
        self.target = target
        self.crop_size=crop_size
        self.rotate=rotate

        #Uncomment and modify args if using source pkl file
        # self.min_max_dict = min_max_dict
        
        if self.stat=='min_max':
            self.min = np.stack([self.stat_dict[var]['min'] for var in input_vars])
            # print(f'self.min shape is {self.min.shape}') # Shape is (NumofInputVars,)
            # print(f'self.min is {self.min}')
            self.max = np.stack([self.stat_dict[var]['max'] for var in input_vars])
        
        if self.stat=='mean_std':
            self.mean = np.stack([self.stat_dict[var]['mean'] for var in input_vars])
            self.std = np.stack([self.stat_dict[var]['std'] for var in input_vars])


    def __len__(self):
        # Return Number of Samples in the Dataset
        return len(self.sample_list)

    def __getitem__(self,idx):

        #fetch a single netcdf file dataset
        sample = self.sample_list[idx].isel(time=-1) #(64,64,Number of Features)

        #make ignition point values binary
        sample['ignition_points'] = sample['ignition_points'].where(sample['ignition_points'] == 0, 1)

        #make burned area values binary
        sample['burned_areas']=sample['burned_areas'].where(sample['burned_areas'] == 0, 1)

        #convert xarray dataset to xarray dataarray
        sample=sample.to_array()

        #Transpose to expected dimensions in model
        
        
        #Uncomment for checking
        #print(f'Before Transpose-shape of sample data array {sample.shape}') # (NumAllVars, 64, 64)
        
        sample=sample.transpose('x','y','variable')
        
       # print(f'shape of sample data array {sample.shape}') #(64,64,NumAllVars)

        #Define target 
        target = sample.sel(variable=self.target).values
        #print(f'Before expanding dims - target shape is {target.shape} and target type is {type(target)}') #(64, 64)

        # Add a dimension to match shape
        target=np.expand_dims(target,axis=-1)
        #print(f'After expanding dims - target shape is {target.shape} and target type is {type(target)}') #(64, 64,1)

        # Select only features in inputvars and stack along the 'variables' axis
        inputs = np.stack([sample.sel(variable=var) for var in self.input_vars],axis=-1).astype(np.float32)
        #print(f'inputs shape is {inputs.shape} and inputs type is {type(inputs)}') # (64, 64, NumInputVars)

        # Normalise all feature values except landcover fractions, binary values such as ignition points and burned areas
        for i, var in enumerate(self.input_vars):
            if not var.startswith('lc_') and (var != 'ignition_points'):
                if self.stat=='min_max':
                    
                    # #Uncomment these lines to perform a sanity check
                    # print(f'shape of inputs[i] is {inputs[:,:,i].shape}') #(64, 64)
                    # print(self.min[i],self.max[i])
                    # print(inputs[:,:,i])

                    inputs[:,:,i] = (inputs[:,:,i] - self.min[i]) / (self.max[i] - self.min[i])

                    # print(inputs[:,:,i])
                elif self.stat=='mean_std':
                    # print(inputs.astype(np.float32))
                    inputs[:,:,i] = (inputs[:,:,i]  - self.mean[i]) / (self.std[i]+1e-10)
                    # print(inputs.astype(np.float32))



        inputs = np.nan_to_num(inputs, nan=0.0)
        target = np.nan_to_num(target, nan=0.0)


        # define lenx and leny as the grid dimensions
        len_x = inputs.shape[0]
        len_y = inputs.shape[1]
        #print(f'len_x is {len_x} and len-y is {len_y}')

        ''' Execute the below statements to randomly crop - Recommended for 64 x 64 only'''
        if self.crop_size > 0:
                    start_x = random.randint(0, len_x - self.crop_size)
                    start_y = random.randint(0, len_y - self.crop_size)
                    end_x = start_x + self.crop_size
                    end_y = start_y + self.crop_size
                    inputs = inputs[start_x:end_x, start_y:end_y,:]
                    target = target[start_x:end_x, start_y:end_y,:]


        # Apply random rotation augmentation
        if self.rotate:
            k = random.randint(0, 3)  # Randomly choose between 0, 1, 2, or 3 90-degree rotations
            inputs = np.rot90(inputs, k, axes=(0, 1))  # Rotate on the first two axes (x, y)
            target = np.rot90(target, k, axes=(0, 1))
        
        inputs=torch.tensor(inputs.copy(), dtype=torch.float32)
        target=torch.tensor(target.copy(), dtype=torch.float32)

        inputs=inputs.permute(2,0,1)
        target=target.permute(2,0,1)
        # print(f'final inputs shape  and type is {type(inputs)} and {inputs.shape} \n  final targets shape and type is {type(target)}  and {target.shape}')
        return inputs,target
    
