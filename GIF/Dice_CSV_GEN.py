import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
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


start = time.time()

# Absolute path to data folder
pa = "C:\\Users\\bedox\\Desktop\\DATA SETS\\Brats2021"
Actual_list = sorted(glob.glob(pa + '/labelsTr/*'))
Predicted_list = sorted(glob.glob(pa + '/Predictions/*'))

Case_list = []
Dice_list = []


for i in range(len(Actual_list)):

    case_num = Predicted_list[i].split("_")[1].split(".")[0]
    case = "Case " + str(case_num)
    Case_list.append(case)
    print(case)

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
    Dice_list.append(dice)


Case_dict = dict(zip(Case_list, Dice_list))
print(Case_list)
print(Dice_list)
print(Case_dict)

with open('C:\\Users\\bedox\\Desktop\\DATA SETS\\Brats2021\\Dice_scores.csv', 'w', newline='') as csvfile:
    field_names = ['Patient', 'Dice Score']
    writer = csv.DictWriter(csvfile, fieldnames = field_names)
    writer.writeheader()
    for key in Case_dict:
        writer.writerow({'Patient': key, 'Dice Score': Case_dict[key]})

end = time.time()
print(end - start)























