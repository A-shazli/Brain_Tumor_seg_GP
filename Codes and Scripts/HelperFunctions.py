import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

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

