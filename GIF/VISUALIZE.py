import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.animation as animate
from imutils import rotate as rot
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import csv
import time
import glob
plt.style.use('dark_background')
import matplotlib as mpl
mpl.rcParams[
        'animation.ffmpeg_path'] = r'C:\Users\bedox\Desktop\ffmpeg-2022-10-27-git-00b03331a0-essentials_build\bin\ffmpeg.exe'



def dice_coefficient(y_true, y_pred, axis=(1, 2, 3),
                     epsilon=0.00001):
    """
    Compute mean dice coefficient over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.
    """

    dice_numerator = 2 * tf.keras.backend.sum(y_true * y_pred , axis = axis) + epsilon
    dice_denominator = tf.keras.backend.sum(y_true, axis = axis ) + tf.keras.backend.sum(y_pred, axis = axis) + epsilon
    dice_coefficient = tf.keras.backend.mean(dice_numerator/dice_denominator)

    return dice_coefficient

def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return dice

scaler = MinMaxScaler()
# Absolute path to data folder
pa = "C:\\Users\\bedox\\Desktop\\DATA SETS\\Brats2021"
# This is in case we want to generate gifs for the whole data
# Vol_list = sorted(glob.glob(pa + '/imagesTR/*_0002.nii.gz'))
# Actual_list = sorted(glob.glob(pa + '/lab/*'))
# Predicted_list = sorted(glob.glob(pa + '/pred/*'))

Vol_list = sorted(glob.glob(pa + '/img/*'))
Actual_list = sorted(glob.glob(pa + '/lab/*'))
Predicted_list = sorted(glob.glob(pa + '/pred/*'))

for i in range(len(Actual_list)):

    # case_num = Predicted_list[i].split("_")[1].split(".")[0]
    case = "Case " + str(i + 1)
    print(case)

    Volume = nib.load(Vol_list[i]).get_fdata()
    Volume = scaler.fit_transform(Volume.reshape(-1, Volume.shape[-1])).reshape(Volume.shape)
    GT_seg = np.around(nib.load(Actual_list[i]).get_fdata())
    seg_pred = np.around(nib.load(Predicted_list[i]).get_fdata())

    # THESE LINES ARE FOR WHEN THE DATA IS NOT BRATS convention
    # GT_seg[GT_seg == 4] = 3
    # GT_seg[GT_seg == 2] = 1
    # GT_seg[GT_seg == 1] = 2
    GT_seg = tf.keras.utils.to_categorical(GT_seg, num_classes=4)

    # Pred[Pred == 4] = 3
    # will remove this as of the new model
    seg_new = np.zeros_like(seg_pred)
    seg_new[seg_pred == 2] = 1
    seg_new[seg_pred == 1] = 2
    seg_new[seg_pred == 3] = 3

    Pred = tf.keras.utils.to_categorical(seg_new, num_classes=4)

    GT_seg = tf.convert_to_tensor(GT_seg)
    Pred = tf.convert_to_tensor(Pred)

    GT_seg = np.moveaxis(GT_seg, -1, 0)
    Pred = np.moveaxis(Pred, -1, 0)

    # fig, axes = plt.subplots(1, 2)
    dice = dice_coef(GT_seg[1:, :, :, :], Pred[1:, :, :, :]).numpy()

    # Slice = random.randint(60, 100)
    # fig.suptitle("\n DICE SCORE: " + str(dice) + "\n Showing slice: {}".format(Slice))

    # print("Showing slice {}".format(Slice))

    GT_seg = np.moveaxis(GT_seg, 0, -1)
    Pred = np.moveaxis(Pred, 0, -1)

    Pred_labeled = np.zeros_like(Pred[:, :, :, 1:])
    Pred_labeled[:, :, :, 0] = Volume * (Pred[:, :, :, 0])
    Pred_labeled[:, :, :, 1] = Volume * (Pred[:, :, :, 0])
    Pred_labeled[:, :, :, 2] = Volume * (Pred[:, :, :, 0])
    Pred_labeled += Pred[:, :, :, 1:]

    GT = np.zeros_like(GT_seg[:, :, :, 1:])
    GT[:, :, :, 0] = Volume * (GT_seg[:, :, :, 0])
    GT[:, :, :, 1] = Volume * (GT_seg[:, :, :, 0])
    GT[:, :, :, 2] = Volume * (GT_seg[:, :, :, 0])
    GT += GT_seg[:, :, :, 1:]

    figure, ax = plt.subplots(1, 3, figsize=(8, 5))
    figure.tight_layout()
    figure.suptitle("\n\nDICE SCORE: " + str(dice))
    ax[0].set_title("GROUND TRUTH")
    ax[1].set_title("PREDICTION")
    ax[2].set_title("Original Volume")
    images = []
    for s in range(Volume.shape[2]):  # gets the number of slices to iterate

        im_gt = ax[0].imshow(rot(GT[:, :, s, :], angle=270), animated=True)
        im_pred = ax[1].imshow(rot(Pred_labeled[:, :, s, :], angle=270), animated=True)
        im = ax[2].imshow(rot(Volume[:, :, s], angle=270), animated=True, cmap="gray")

        t = ax[0].text(5, 22, f"Slice: {s}", fontsize="smaller")  # add text
        images.append([im, im_gt, im_pred, t])

    ani = animate.ArtistAnimation(figure, images, interval=50, \
                                       blit=True, repeat_delay=500)

    ani.save("C:\\Users\\bedox\\Desktop\\Dice_GIF\\" + "Prediction VS Ground Truth_case " + str(i + 1) + " axial view.gif")
    print("\n Starting Case " + str(i + 1) + " done....")


    # axes[0].set_title("GROUND TRUTH")
    # axes[0].imshow(rot(GT[:, :, Slice, :], angle=270))
    # axes[1].set_title("PREDICTION")
    # axes[1].imshow(rot(Pred_labeled[:, :, Slice, :], angle=270))

    # plt.show()


