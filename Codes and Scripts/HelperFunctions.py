import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import nibabel as nib
import json
import os
import shutil

def preprocess_image_train(image):
    image = (image/127.5)-1
    return image

# TO DO: Generalize Path, add it to arguments
def generate_images_GIF(img_input, model, img_true, mode, order):
    prediction = model(img_input)
    pred_vol = prediction[0, :, :, 0].numpy().copy()
    error = tf.image.ssim(img_true, prediction, max_val=2)
    img_input = np.rot90(img_input[0, :, :, 0], 3)
    img_true = np.rot90(img_true[0, :, :, 0], 3)
    prediction = np.rot90(prediction[0, :, :, 0], 3)

    plt.figure(figsize=(10, 6))
    if mode == 1:
        display_list = [img_input, prediction, img_true]
        title = [f'{seq_1} True', f'{seq_2} predicted', f'{seq_2} True']

    else:
        display_list = [img_input, prediction, img_true]
        title = [f'{seq_2} True', f'{seq_1} predicted', f'{seq_1} True']

    plots_path_T1_FLAIR = r'E:\Graduation Project\GIFs and Models\Brats {}\Predicted\{}-{}-GIF'.format(brats_num, seq_1,
                                                                                                       seq_2)
    plots_path_FLAIR_T1 = r'E:\Graduation Project\GIFs and Models\Brats {}\Predicted\{}-{}-GIF'.format(brats_num, seq_2,
                                                                                                       seq_1)
    if not os.path.exists(plots_path_T1_FLAIR):
        os.makedirs(plots_path_T1_FLAIR)
    if not os.path.exists(plots_path_FLAIR_T1):
        os.makedirs(plots_path_FLAIR_T1)

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
        if mode == 1:
            plt.savefig(
                r'E:\Graduation Project\GIFs and Models\Brats {}\Predicted\{}-{}-GIF\{}.png'.format(brats_num, seq_1,
                                                                                                    seq_2, order))
        if mode == 2:
            plt.savefig(
                r'E:\Graduation Project\GIFs and Models\Brats {}\Predicted\{}-{}-GIF\{}.png'.format(brats_num, seq_2,
                                                                                                    seq_1, order))
    plt.show()
    return error, pred_vol


#TO DO: Generalize Path, add it to arguments
def predict_image(img_input, model):
    prediction = model(img_input)
    pred_vol = prediction[0, :, :, 0].numpy().copy()
    return pred_vol

#TO DO: Generalize Path, add it to arguments
def predict_image_and_calc_loss(img_input, model, img_true):
    prediction = model(img_input)
    pred_vol = prediction[0, :, :, 0].numpy().copy()
    error = tf.image.ssim(img_true, prediction, max_val=2)
    return error, pred_vol

def black_seq_generator(test_path, brats_num, T1_FLAG=True, T2_FLAG=True, FLAIR_FLAG=True):
    test_data_list = sorted(glob.glob(test_path + '/*'))
    original_vol_path = sorted(glob.glob(test_path + '/*'))[0]
    original_vol = nib.load(original_vol_path)
    original_shape = original_vol.shape

    v = np.zeros(original_shape)
    v = nib.Nifti1Image(v, original_vol.affine)  # to save this 3D (ndarry) numpy

    if FLAIR_FLAG:
        nib.save(v, test_path + '/' + f'BraTS2021_0{brats_num:04d}_flair.nii.gz')
    if T1_FLAG:
        nib.save(v, test_path + '/' + f'BraTS2021_0{brats_num:04d}_t1.nii.gz')
    if T2_FLAG:
        nib.save(v, test_path + '/' + f'BraTS2021_0{brats_num:04d}_t2.nii.gz')

    if T1_FLAG and T2_FLAG:
        nib.save(v, test_path + '/' + f'BraTS2021_0{brats_num:04d}_t1.nii.gz')
        nib.save(v, test_path + '/' + f'BraTS2021_0{brats_num:04d}_t2.nii.gz')
    if T1_FLAG and FLAIR_FLAG:
        nib.save(v, test_path + '/' + f'BraTS2021_0{brats_num:04d}_t1.nii.gz')
        nib.save(v, test_path + '/' + f'BraTS2021_0{brats_num:04d}_flair.nii.gz')
    if T2_FLAG and FLAIR_FLAG:
        nib.save(v, test_path + '/' + f'BraTS2021_0{brats_num:04d}_flair.nii.gz')
        nib.save(v, test_path + '/' + f'BraTS2021_0{brats_num:04d}_t2.nii.gz')
    print("Done")

def copy_subfolders_into_another_folder(paths_txt, source_folder, destination_folder):
    # Read the contents of the file
    with open(paths_txt, 'r') as file:
        contents = file.read()

    # Replace single quotes with double quotes
    contents = contents.replace("'", "\"")

    # Load the JSON array, the list of subfolder names to copy
    subfolder_names = json.loads(contents)

    # Loop through each item in the source folder
    for item in os.listdir(source_folder):
        # If the item is a subfolder and its name is in the list
        if os.path.isdir(os.path.join(source_folder, item)) and item in subfolder_names:
            # Copy the subfolder to the destination folder
            shutil.copytree(os.path.join(source_folder, item), os.path.join(destination_folder, item))

    copy_subfolders_into_another_folder

# The following function is responsible for returning the indices of the brain of the volume that contains foreground voxels.
def find_brain_width_wise(dep, hei, i, img):        #cropping width wise
    slice2D = img.get_fdata()[:, i, :]
    for j in range(hei):
        for k in range(dep):
            if slice2D[j, k] != 0:
                return i
    return 0

def find_brain_height_wise(dep, wid, i, img):      #cropping height wise
    slice2D = img.get_fdata()[i, :, :]
    for j in range(wid):
        for k in range(dep):
            if slice2D[j, k] != 0:
                return i
    return 0

def find_brain_depth_wise(wid, hei, i, img):        #cropping depth wise
    slice2D = img.get_fdata()[:, :, i]
    for j in range(wid):
        for k in range(hei):
            if slice2D[j, k] != 0:
                return i
    return 0

# def brain_cropper_with_indices()

