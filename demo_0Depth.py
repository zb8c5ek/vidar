from PIL import Image
from hubconf import ZeroDepth
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""


def tensor_to_numpy_image(tensor_in):
    """
    Convert a BCHW or CHW PyTorch tensor to a BHWC or HWC NumPy array.
    Assumes input tensor might be on GPU, might have data in range [0,1] or [-1,1].
    Outputs a NumPy array with data in range [0,255].
    """
    # Make sure it's on CPU
    tensor = tensor_in.cpu().detach().squeeze()
    # Convert to numpy array
    img = tensor.numpy()

    return img


def visualize_images_from_model(img_rgb, depth_pred):
    """
    Visualizes the RGB and depth prediction images side by side.

    Parameters:
    - img_rgb (np.array): RGB image
    - depth_pred (np.array): Depth prediction image
    """
    plt.clf()
    if img_rgb.shape[:2] != depth_pred.shape[:2]:
        raise ValueError("Both images should have the same resolution.")

    # Set up a figure with correct aspect ratio and size
    fig_width_inches = 20  # for example, adjust if needed
    fig_height_inches = fig_width_inches * img_rgb.shape[0] / img_rgb.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2 * fig_width_inches, fig_height_inches))

    # Display RGB image
    axes[0].imshow(img_rgb)
    axes[0].axis('off')
    axes[0].set_title('RGB Image')

    # Display depth prediction with a colorbar
    im = axes[1].imshow(depth_pred, cmap='viridis')  # Using viridis colormap, but you can change it
    axes[1].axis('off')
    axes[1].set_title('Depth Prediction')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)  # Attach the colorbar

    # Ensure layout is tight so images are side by side without much space
    plt.tight_layout()
    plt.show()
    plt.close()


# You can now use this function as:
# visualize_images(img_rgb, depth_pred)


if __name__ == '__main__':
    # zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)
    zerodepth_model = ZeroDepth(pretrained=True)

    intrinsics = torch.tensor(np.load('examples/ddad_intrinsics.npy')).unsqueeze(0)

    dp_images = Path("E:\sfm\data\pco_image_front_30_straight\images").resolve()
    fps_img = sorted(dp_images.glob("*.jpg"))
    if torch.cuda.is_available():
        zerodepth_model = zerodepth_model.cuda()
    zerodepth_model.eval()
    # DOING: THEY ARE AT FIXED DEPTH, MAYBE JUST USE THEM TO INITIALIZE THE COMPUTATION

    img_net_shape = tuple((np.array([1920, 1080]) * 0.5).astype(int))  # Image use X-Y coordinates, #Columns then #Rows
    # 1080P IMAGE SEEMS OK, WITH 11.6G GMEM AND 18S INFERENCE TIME, but results is not good
    # 0.5 gives good results, and 6S INFERENCE TIME
    # 0.6 GIVES BLURRED RESULTS.
    # 0.3 GIVES BLURRED RESULTS AS WELL
    with torch.no_grad():
        for fp in tqdm(fps_img):
            img_ori = Image.open(fp)
            if img_ori.size[:2] != img_net_shape:
                img = np.array(img_ori.resize(img_net_shape))
            else:
                img = np.array(img_ori)
            img_for_model = torch.tensor(img).permute(2, 0, 1).unsqueeze(0) / 255.
            # img_rgb = img_rgb[:, :, ::4, ::4]
            # rgb = torch.tensor(cv2.imread('examples/ddad_sample.png')).permute(2, 0, 1).unsqueeze(0) / 255.
            if torch.cuda.is_available():
                img_for_model = img_for_model.cuda()
                intrinsics = intrinsics.cuda()
            depth_pred = zerodepth_model(img_for_model, intrinsics)
            visualize_images_from_model(img, tensor_to_numpy_image(depth_pred))
