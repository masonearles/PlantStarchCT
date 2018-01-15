# Import libraries

import ijroi
from sima.ROI import ROI, ROIList
from skimage import io, img_as_ubyte
import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import median, sobel, hessian, gabor, gaussian, scharr
import cv2
from skimage.transform import rotate
import ijroi
from tqdm import tqdm
from sima.ROI import ROI, ROIList
import os
from scipy import ndimage
from astropy.convolution import convolve, Gaussian2DKernel, interpolate_replace_nans
from scipy.ndimage.filters import median_filter, minimum_filter, maximum_filter


# Import full RAP ROIs
def ROILoadFromDir(_path,_dir):   
    fullRAP_roi_list = []
    for i in range(0,len(_dir)):
        fullRAP_roi_list.append(
            ROIList.load(_path + _dir[i], fmt='ImageJ')
        )
    return fullRAP_roi_list
    

# Set im_shape for all ROIs based on CT shape
def SetImgShp(roi,CT_tif): 
    for i in range(0,len(roi)-1):
        roi[i].im_shape = CT_tif.shape
    return roi


# Convert ImageJ ROI polygons into list of 2D coordinates
def ConvROItoList(roi):
    roi_list = []
    for j in range(0,len(roi)):
        roi_array = np.zeros((len(roi[j].coords[0]),2))
        for i in range(0,len(roi[j].coords[0])):
            roi_array[i,:] = roi[j].coords[0][i][[0,1]]
        roi_list.append(roi_array)
    return roi_list


# Generate binary mask with shape of tiff using ROI
def GenROIMask(roi_list,CT_tif):
    nx, ny = CT_tif.shape
    img = Image.new("F", [nx, ny], 0)
    for i in range(0,len(roi_list)):
        poly = tuple(map(tuple, roi_list[i]))
        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    mask = np.array(img)
    return mask

# Center and scale CT images
def CenterScaleCT(CT_tif):
    CT_tif_centerscale = [(CT_tif[i]-CT_tif[i].mean())/CT_tif[i].std() for i in range(0,len(CT_tif))]
    return CT_tif_centerscale

# Generate manually labeled array of starch full and empty RAP regions
def GenLabelImg(labelRAP_mask,label_mask):
    label_img = labelRAP_mask + label_mask
    return label_img

# Variance filter function
def winVar(img, wlen):
    box_kernel = np.ones((wlen,wlen))
    wmean, wsqrmean = (convolve(x, box_kernel, boundary = 'extend', nan_treatment = 'interpolate')
     for x in (img, img*img))
    return wsqrmean - wmean*wmean


# Membrane projection filter (similar to ImageJ Weka)
# Requires three inputs: 1) image to filter, 2) membrane length, and 3) membrane width
# This is an attempt to capture the cell walls within empty regions of RAPs, so
# use the verage cell length and cell wall width as l_mem and w_mem, respectively.
# E.g. one of the Riparia scans appears to have l_mem ~= 11 px and w_mem ~= 2 px.
# This filter works by generating a convolution kernel of width = l_mem. A line is then
# added to the kernel. This line is rotated by 6-deg increments generating a total of 30 kernels.
# These 30 kernels are then convolved with the tif image and a z-projection is done with six operations:
# 1) sum, 2) mean, 3) variance, 4) median, 5) max, and 6) min. 
def MembraneFilter(img,l_mem,w_mem):
    nx = ny = l_mem
    n_mid = (nx-1)/2
    zero_array = np.zeros((nx,ny))
    rot_img = np.zeros((nx,ny))
    img_conv = np.zeros((img.shape[0],img.shape[1],30))
    for i in range(0,30):
        zero_array[:,(n_mid-w_mem):(n_mid+w_mem)] = 1
        rot_img[:,:] = rotate(zero_array,i*6)
        img_conv[:,:,i] = convolve(img, rot_img, boundary = 'extend', nan_treatment = 'interpolate')
    
    return np.dstack((np.nansum(img_conv,2), np.nanmean(img_conv,2), np.nanvar(img_conv,2), np.nanmedian(img_conv,2), np.nanmax(img_conv,2), np.nanmin(img_conv,2)))


def PatchFilter(img,wlen,dotlen):
    nx = ny = wlen
    zero_array = np.negative(np.ones((nx,ny)))
    boundlow = (wlen-dotlen)/2
    boundhigh = wlen-(wlen-dotlen)/2
    zero_array[boundlow:boundhigh,boundlow:boundhigh] = wlen
    img_conv = [convolve(img[i], zero_array,boundary = 'extend', nan_treatment = 'interpolate')
     for i in range(0,len(img))]
    return img_conv


def MedianFilter(img,wlen):
    img_conv_med = [median_filter(interpolate_replace_nans(img[i],np.ones((29,29))),
                                 size = wlen) for i in range(0,len(img))]
    return img_conv_med


def MinFilter(img,wlen):
    img_conv_min = [minimum_filter(interpolate_replace_nans(img[i],np.ones((29,29))),
                                   size = wlen) for i in range(0,len(img))]
    return img_conv_min


def MaxFilter(img,wlen):
    img_conv_max = [maximum_filter(interpolate_replace_nans(img[i],np.ones((29,29))),
                                   size = wlen) for i in range(0,len(img))]
    return img_conv_max


def VarianceRangeFilter(img,wlen_tup):
    b = []
    for i in range(0,len(img)):
        a = np.zeros((img[i].shape[0],img[i].shape[1],len(wlen_tup)))
        for j in range(0,len(wlen_tup)):
            a[:,:,j] = winVar(img[i],wlen_tup[j])
        b.append(a)
    VRF_sum = [np.nansum(b[i],2) for i in range(0,len(b))]
    VRF_min = [np.nanmin(b[i],2) for i in range(0,len(b))]
    VRF_max = [np.nanmax(b[i],2) for i in range(0,len(b))]
    VRF_mean = [np.nanmean(b[i],2) for i in range(0,len(b))]
    return VRF_sum, VRF_min, VRF_max, VRF_mean


# Feature layer generation from 32-bit grid reconstructed microCT image
def GenFeatureLayers(CT_tif,wlen_tup):
    
    # MF_tmp = map((lambda i: MembraneFilter(CT_tif[i],l_mem,w_mem)), range(0,len(CT_tif)))
    
    # Generate patch filtered stack
    #PF = PatchFilter(CT_tif,7,3)
    
    # Generate VarianceRangeFiltered stack
    VRF_sum, VRF_min, VRF_max, VRF_mean = VarianceRangeFilter(CT_tif,wlen_tup)
    
    # Generate Gaussian kernels
    GaussianKernels = []
    GaussianKernels.append([Gaussian2DKernel(stddev=i, x_size=3*i, y_size=3*i) for i in (1,3,5)])
    
    # Define feature layer array dimensions
    nFL = 11 # Number of feature layers (count them up below)
    nPlants = len(CT_tif) # Number of plants used for training
    
    # Define empty numpy array for feature layers (FL)
    FL_list = []
    
    # Populate FL array with feature layers using custom filters, etc.
    for i in tqdm(range(0,nPlants)):
        nx, ny = CT_tif[i].shape # CT image dimensions
        FL = np.empty((nx,ny,nFL), dtype=np.float64)
        FL[:,:,0] = CT_tif[i][:,:]
        FL[:,:,1] = convolve(FL[:,:,0],GaussianKernels[0][0])
        FL[:,:,2] = convolve(FL[:,:,0],GaussianKernels[0][1])
        FL[:,:,3] = convolve(FL[:,:,0],GaussianKernels[0][2])
        FL[:,:,4] = VRF_sum[i]
        FL[:,:,5] = VRF_min[i]
        FL[:,:,6] = VRF_max[i]
        FL[:,:,7] = VRF_mean[i]
        FL[:,:,8] = convolve(FL[:,:,5],GaussianKernels[0][0])
        FL[:,:,9] = convolve(FL[:,:,5],GaussianKernels[0][1])
        FL[:,:,10] = convolve(FL[:,:,5],GaussianKernels[0][2])
        FL_list.append(FL)   
    
        
    # Collapse training data to two dimensions for Random Forest training algorithm
    #FL_reshape = FL.reshape((-1,FL.shape[3],FL.shape[0]), order="F")
    return FL_list


def SavePredictions_ListToTif(path,CT_dir,img):
    img_8bit = [img_as_ubyte(np.divide(img[i],float(2))) for i in range(0,len(img))]
    for i in range(0,len(img)):
        io.imsave(path + 'PredictedLabel_' + CT_dir[i], img_8bit[i])


# Reshape, flip and rotate
def RFR(FL,img_for_shape):
    nFL = 21 # Number of feature layers (count them up above)
    nPlants = 10 # Number of plants
    #FL_RFR = np.rot90(np.flip(FL.reshape(nPlants, img_for_shape[0].shape[0],
                                         #img_for_shape[0].shape[1],nFL),1),1,(1,2))
    FL_RFR = np.rot90(FL.reshape(nPlants, img_for_shape[0].shape[0],
                                         img_for_shape[0].shape[1],nFL),1,(1,2))
    return(FL_RFR)

# Apply binary mask to 32-bit CT image
def ApplyROIMask(mask, CT_tif):
    CT_tif_mask = CT_tif*mask
    return CT_tif_mask