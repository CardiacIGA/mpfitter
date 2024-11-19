from pathlib import Path
from typing import Tuple

import numpy as np
import PIL
import SimpleITK
from PIL.Image import Resampling

###########################################
# PARAMETERS TO PLAY WITH

# Select the patient identification (scalar value between 1 and 98)
patient_id = 3


###########################################
# Definition of useful functions

def load_mhd(filepath: Path) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Loads a mhd image and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - (dH,dW), Size of the voxels along the (height, width) dimension (in mm).
    """
    # load image and save info
    image = SimpleITK.ReadImage(str(filepath))
    info = (image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection())

    # create numpy array from the .mhd file and corresponding image
    im_array = np.squeeze(SimpleITK.GetArrayFromImage(image))

    # extract voxelspacing from image metadata
    info = [item for sublist in info for item in sublist]
    voxelspacing = info[6:8][::-1]

    return im_array, voxelspacing

def resize_image(image: np.ndarray, size: Tuple[int, int], resample: Resampling = Resampling.NEAREST) -> np.ndarray:
    """Resizes the image to the specified dimensions.

    Args:
        image: Input image to process. Must be in a format supported by PIL.
        size: Width and height dimensions of the processed image to output.
        resample: Resampling filter to use.

    Returns:
        Input image resized to the specified dimensions.
    """
    resized_image = np.array(PIL.Image.fromarray(image).resize(size, resample=resample))
    return resized_image

def resize_image_to_isotropic(image: np.ndarray, spacing: Tuple[float, float]) -> np.ndarray:
    """Resizes an image to have isotropic spacing by downscaling the height.

    Args:
        image: (H, W), Image array.
        spacing: (dH, dW) Anistropic spacing.

    Returns:
        Image, downsampled on the height dimension, with isotropic spacing.
    """
    real_aspect = (image.shape[0] * spacing[0]) / (image.shape[1] * spacing[1])
    current_aspect = (image.shape[0]) / (image.shape[1])
    new_height = int(image.shape[0] * (real_aspect / current_aspect))
    new_width = image.shape[1]
    return resize_image(image, (new_width, new_height))


# Specify the ultrasound/segmentation pair to be loaded
patient_name = f"patient{patient_id:03d}"
patient_dir = Path(f"testing/input/{patient_name}")
path_output = Path(f"testing/output/")
path_to_bmode_images = patient_dir / f"{patient_name}_4CH_sequence.mhd"
path_to_gt_segmentations = patient_dir / f"{patient_name}_4CH_sequence_gt.mhd"
print(f"Loading data from patient folder: {patient_dir}")

# Call of a specific function that reads the .mhd files and gives access to the corresponding images and metadata
bmode, voxelspacing = load_mhd(path_to_bmode_images)
gt, _ = load_mhd(path_to_gt_segmentations)
nb_frames, width, height = bmode.shape


# Display the corresponding useful information
print(f"{type(bmode)=}")
print(f"{bmode.dtype=}")
print(f"{bmode.shape=}")
print('')

print(f"{type(gt)=}")
print(f"{gt.dtype=}")
print(f"{gt.shape=}")


# Resize the ultrasound/segmentation pair to have isotropic voxelspacing
bmode = np.array([resize_image_to_isotropic(bmode_2d, voxelspacing) for bmode_2d in bmode])
gt = np.array([resize_image_to_isotropic(gt_2d, voxelspacing) for gt_2d in gt])


# Display the corresponding useful information
print(f"{type(bmode)=}")
print(f"{bmode.dtype=}")
print(f"{bmode.shape=}")
print('')

print(f"{type(gt)=}")
print(f"{gt.dtype=}")
print(f"{gt.shape=}")


# Display the points on the corresponding image
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

px = 1/plt.rcParams['figure.dpi']  # pixel in inches

fig = plt.figure(figsize=(width*px, height*px))
bmode_im = plt.imshow(bmode[0], cmap='gray')
gt_im = plt.imshow(np.ma.masked_where(gt[0] == 0, gt[0]), interpolation='none', cmap='jet', alpha=0.5)
plt.axis("off")
plt.tight_layout()
plt.close() # this is required to not display the generated image

def init():
    """Function that initializes the first frame of the video"""
    bmode_im.set_data(bmode[0])
    gt_im.set_data(gt[0])

def animate(frame_idx):
    """Callback that fetches the data for subsequent frames."""
    bmode_im.set_data(bmode[frame_idx])
    gt_im.set_data(np.ma.masked_where(gt[frame_idx] == 0, gt[frame_idx]))
    return bmode_im, gt_im

interval = 5000 / nb_frames # Adjust delay between frames so that animation lasts 5 seconds
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(bmode), interval=interval)
#HTML(anim.to_html5_video())
anim.save(path_output+'SegmentedECHO_combatVT.mp4', writer = 'ffmpeg', fps = 30)
