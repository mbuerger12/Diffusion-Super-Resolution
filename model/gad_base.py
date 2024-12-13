import os
from random import randrange
from re import I
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

INPUT_DIM = 4
FEATURE_DIM = 64

class GADBase(nn.Module):
    
    def __init__(
            self, feature_extractor='UNet',
            Npre=8000, Ntrain=1024, 
    ):
        super().__init__()

        self.feature_extractor_name = feature_extractor    
        self.Npre = Npre
        self.Ntrain = Ntrain
 
        if feature_extractor=='none': 
            # RGB verion of DADA does not need a deep feature extractor
            self.feature_extractor = None
            self.Ntrain = 0
            self.logk = torch.log(torch.tensor(0.03))

        elif feature_extractor=='UNet':
            # Learned verion of DADA
            self.feature_extractor =  torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bicubic'),
                smp.Unet('resnet50', classes=FEATURE_DIM, in_channels=INPUT_DIM),
                torch.nn.AvgPool2d(kernel_size=2, stride=2) ).cuda()
            self.logk = torch.nn.Parameter(torch.log(torch.tensor(0.03)))

        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')

    def plot_tensor_image(self, img_tensor, path, title="Image", cmap="viridis", slice_idx=0, ):
        """
        Plots the given image tensor.

        Parameters:
            img_tensor (torch.Tensor): The tensor to plot. Shape can be
                (N, C, H, W), (C, H, W), (H, W), or (1, C, H, W).
            title (str): Title of the plot.
            cmap (str): Colormap for grayscale images (default: 'viridis').
            slice_idx (int): The index of the slice to plot if the input has multiple slices (default: 0).
        """
        # Handle batch dimension (N, C, H, W) or (1, C, H, W)
        if len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1:  # Single batch
            img_tensor = img_tensor[0]  # Remove batch dimension

        if len(img_tensor.shape) == 4:  # Batch of channels (C, H, W)
            # Select the specified slice along the channel dimension
            if slice_idx < 0 or slice_idx >= img_tensor.shape[0]:
                raise ValueError(f"Invalid slice_idx {slice_idx} for tensor with shape {img_tensor.shape}")
            img_tensor = img_tensor[slice_idx]  # Select the desired channel

        # Move tensor to CPU and convert to NumPy
        img = img_tensor.detach().cpu().numpy()

        # Handle different shapes
        if len(img.shape) == 3:  # Multi-channel image (C, H, W)
            img = img.transpose(1, 2, 0)  # Convert to (H, W, C)
            if img.shape[2] == 1:  # Single channel, convert to 2D
                img = img.squeeze(-1)

        elif len(img.shape) != 2:  # If not (H, W) or (H, W, C), raise error
            raise ValueError(f"Unsupported tensor shape after processing: {img_tensor.shape}")

        # Normalize image for display if needed
        if img.max() > 1 or img.min() < 0:
            img = (img - img.min()) / (img.max() - img.min())

        # Plot the image
        plt.figure(figsize=(6, 6))
        if len(img.shape) == 2:  # Grayscale image
            plt.imshow(img, cmap=cmap)
        else:  # RGB image
            plt.imshow(img)
        plt.title(title)
        plt.axis("off")

        save_path = os.path.join(path, title)+".png"
        plt.savefig(save_path)
        plt.show()

    def forward(self, sample, train=False, deps=0.1):
        guide, source, mask_lr = sample['guide'], sample['source'], sample['mask_lr']
        self.sample_name = sample['img_path'].split('\\')[-1]
        # assert that all values are positive, otherwise shift depth map to positives
        print(source.min())
        if source.min()<deps:
            print("Warning: The forward function was called with negative depth values. Values were temporarly shifted. Consider using unnormalized depth values for stability.")
            source += deps
            sample['y_bicubic'] += deps
            shifted = True
        else:
            shifted = False

        y_pred, aux = self.diffuse(sample['y_bicubic'].clone(), guide.clone(), source, sample['y'], mask_lr > 0.5,
                 K=torch.exp(self.logk),  verbose=False, train=train)

        # revert the shift
        if shifted:
            y_pred -= deps

        # return {'y_pred': y_pred} | aux
        return {**{'y_pred': y_pred}, **aux}


    def diffuse(self, img, guide, source, y, mask_inv,
        l=0.24, K=0.01, verbose=False, eps=1e-8, train=False):

        _, _,h,w = guide.shape
        _, _,sh,sw = source.shape

        # Define Downsampling operations that depend on the input size
        downsample = nn.AdaptiveAvgPool2d((sh, sw))
        upsample = lambda x: F.interpolate(x, (h, w), mode='nearest')
        # Deep Learning version or RGB version to calucalte the coefficients
        print(self.feature_extractor_name)
        if self.feature_extractor is None:
            guide_feats = torch.cat([img, guide], 1)
        else:
            guide_feats = self.feature_extractor(torch.cat([img-img.mean((1,2,3), keepdim=True), guide ], 1))
            #guide_feats = guide_feats.permute(0,3,1,2)
        # Convert the features to coefficients with the Perona-Malik edge-detection function
        cv, ch = c(guide_feats, K=K)

        if '359' in self.sample_name:
            # Convert tensors to NumPy
            """
            cvi = cv.cpu().numpy()  # Shape: (1, 2, 511, 511)
            chi = ch.cpu().numpy()  # Shape: (1, 2, 511, 511)

            # Select the first batch and individual channels
            cvi = cvi[0, 0, :, :]  # First channel of cv, shape: (511, 511)
            chi = chi[0, 0, :, :]  # Second channel of ch, shape: (511, 511)

            # Plot cv and ch separately
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Plot cv
            axes[0].imshow(cvi, cmap='viridis')
            axes[0].set_title('Coefficient cv')
            axes[0].axis('off')  # Turn off axes for better visualization

            # Plot ch
            axes[1].imshow(chi, cmap='viridis')
            axes[1].set_title('Coefficient ch')
            axes[1].axis('off')  # Turn off axes for better visualization

            # Adjust layout and show the plots
            plt.tight_layout()
            plt.show()
            """
            # Plot the combined image
            dir_name = os.path.join('save_img_dir', f"epoch_{str(len(os.listdir('save_img_dir')))}")
            os.mkdir(dir_name)
            self.plot_tensor_image(img, title="image", path=dir_name)
            self.plot_tensor_image(guide, title="guide", path=dir_name)
            self.plot_tensor_image(source, title="source", path=dir_name)
            #for i in range(0,FEATURE_DIM):
            self.plot_tensor_image(y, title="label", path=dir_name)
            self.plot_tensor_image(guide_feats[:, 0, :, :], title=f"guide_feats", path=dir_name)
            self.plot_tensor_image(guide_feats[:, 1, :, :], title=f"guide_feats", path=dir_name)
            self.plot_tensor_image(guide_feats[:, 2, :, :], title=f"guide_feats", path=dir_name)
            self.plot_tensor_image(guide_feats[:, 3, :, :], title=f"guide_feats", path=dir_name)
            self.plot_tensor_image(guide_feats[:, 4, :, :], title=f"guide_feats", path=dir_name)
            self.plot_tensor_image(guide_feats[:, 5, :, :], title=f"guide_feats", path=dir_name)



        # Iterations without gradient
        if self.Npre>0:
            with torch.no_grad():
                Npre = randrange(self.Npre) if train else self.Npre
                for t in range(Npre):
                    img = diffuse_step(cv, ch, img, l=l)
                    img = adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8)
        if '359' in self.sample_name:
            self.plot_tensor_image(img, title="image-Npre", path=dir_name)

        # Iterations with gradient
        if self.Ntrain>0:
            for t in range(self.Ntrain):
                img = diffuse_step(cv, ch, img, l=l)
                img = adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8)
        if '359' in self.sample_name:
            self.plot_tensor_image(img, title="image-Ntrain", path=dir_name)
        return img, {"cv": cv, "ch": ch}


# @torch.jit.script
def c(I, K: float=0.03):
    # apply function to both dimensions
    cv = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,1:,:] - I[:,:,:-1,:]), 1), 1), K)
    ch = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,:,1:] - I[:,:,:,:-1]), 1), 1), K)
    return cv, ch

# @torch.jit.script
def g(x, K: float=0.03):
    # Perona-Malik edge detection
    return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))

@torch.jit.script
def diffuse_step(cv, ch, I, l: float=0.24):
    # Anisotropic Diffusion implmentation, Eq. (1) in paper.

    # calculate diffusion update as increments
    dv = I[:,:,1:,:] - I[:,:,:-1,:]
    dh = I[:,:,:,1:] - I[:,:,:,:-1]
    
    tv = l * cv * dv # vertical transmissions
    I[:,:,1:,:] -= tv
    I[:,:,:-1,:] += tv 

    th = l * ch * dh # horizontal transmissions
    I[:,:,:,1:] -= th
    I[:,:,:,:-1] += th 
    
    return I

def adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8):
    # Implementation of the adjustment step. Eq (3) in paper.

    # Iss = subsample img
    img_ss = downsample(img)

    # Rss = source / Iss
    ratio_ss = source / (img_ss + eps)

    mask_inv_broadcasted = mask_inv.expand_as(ratio_ss)
    ratio_ss[mask_inv_broadcasted] = 1

    # R = NN upsample r
    ratio = upsample(ratio_ss)

    # ratio = torch.sqrt(ratio)
    # img = img * R
    return img * ratio 
