import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch


def get_loss(output, sample):
    dir_name = os.path.join('save_img_dir', f"epoch_{str(len(os.listdir('save_img_dir'))-1)}")
    plot_tensor_image(output['y_pred'], dir_name, title="y_pred")
    plot_tensor_image(sample['y'], dir_name, title="y")


    y_pred = output['y_pred']
    color_correction = output['color_correction']
    y, mask_hr, mask_lr = (sample[k] for k in ('y', 'mask_hr', 'mask_lr'))


    loss_color = mse_loss_func(color_correction, y)
    l1_loss = l1_loss_func(y_pred, y)
    mse_loss = mse_loss_func(y_pred, y)
    print(f"l1_loss: {l1_loss}")
    print(f"mse_loss: {mse_loss}")
    loss = mse_loss*100 + loss_color

    return loss, {
        'l1_loss': l1_loss.detach().item(),
        'mse_loss': mse_loss.detach().item(),
        'optimization_loss': loss.detach().item(),
    }
        
#add masks back
def mse_loss_func(pred, gt):
    #return F.mse_loss(pred[mask == 1.], gt[mask == 1.]
    return F.mse_loss(pred, gt)


def l1_loss_func(pred, gt):
    #return F.l1_loss(pred[mask == 1.], gt[mask == 1.])
    return F.l1_loss(pred, gt)



def plot_tensor_image(img_tensor, path, title="Image", cmap="viridis", slice_idx=0, ):
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
        """
        # Normalize image for display if needed
        if img.max() > 1 or img.min() < 0:
            img = (img - img.min()) / (img.max() - img.min())
        """
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