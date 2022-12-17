def gif_gen():
    """
    SEGMENTATION CLASSES SHOULD BE CONSECUTIVE INTEGERS

    This is the main method, it takes in a file bath arranged in this format:

    */*/Data_folder
    ├── imagesTR
    │   ├── XXX_0000.nii.gz
    │   ├── XXX_0001.nii.gz
    │   ├── XXX_0002.nii.gz
    │   ├── XXX_0003.nii.gz
    │   ├── ...
    ├── labelTr
    │   ├── XXX_0000.nii.gz
    │   ├── XXX_0001.nii.gz
    │   ├── ...

    Or you can adjust the paths as needed
    Where XXX can be any name desired what matters is the ids

    Can support up to 4 different MRI modes: 0000, 0001, 0002, 0003 For each patient
    """

    import nibabel as nib
    import glob
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.animation as animate
    from imutils import rotate as rot
    import matplotlib as mpl
    # To be able to generate the gif, download the ffmpeg essentials from here https://www.gyan.dev/ffmpeg/builds/ and
    # Set the target path to that folder
    mpl.rcParams[
        'animation.ffmpeg_path'] = r'C:\Users\bedox\Desktop\ffmpeg-2022-10-27-git-00b03331a0-essentials_build\bin\ffmpeg.exe'
    import tensorflow
    import numpy as np
    import sys

    # if "idlelib" in sys.modules is True:
    #     print("Edit the path variable: pa")
    # else:
    #     pa = input("Please enter path: ")

    # Root path
    pa = "C:\\Users\\bedox\\Desktop\\DATA SETS\\Brats2021"
    classes = input("Please input number of classes: ")

    # Image folder
    list_mode = sorted(glob.glob(pa + '/imagesTR/*'))

    list_mode_1 = []
    list_mode_2 = []
    list_mode_3 = []
    list_mode_4 = []
    all_modes = []

    for path in list_mode:
        if "0000." in path:
            list_mode_1.append(path)
        elif "0001." in path:
            list_mode_2.append(path)
        elif "0002." in path:
            list_mode_3.append(path)
        else:
            list_mode_4.append(path)

    modality = 0

    if len(list_mode_1) != 0:
        all_modes.append(list_mode_1)
        modality += 1
    if len(list_mode_2) != 0:
        all_modes.append(list_mode_2)
        modality += 1
    if len(list_mode_3) != 0:
        all_modes.append(list_mode_3)
        modality += 1
    if len(list_mode_4) != 0:
        all_modes.append(list_mode_4)
        modality += 1

    list_Labels = sorted(glob.glob(pa + '/labelTr/*'))
    fig, axs = plt.subplots(modality, 3)
    plt.style.use('dark_background')
    fig.patch.set_facecolor('black')
    scaler = MinMaxScaler()
    gif = []

    for i in range(len(list_Labels)):
        temp_seg = nib.load(list_Labels[i]).get_fdata().astype(np.uint8)

        # Pdding to make sure all slices end together
        dim = max(temp_seg.shape) // 4
        temp_seg = np.pad(temp_seg, ((((dim * 4 - temp_seg.shape[0]) // 2), ((dim * 4 - temp_seg.shape[0]) // 2)), \
                                     (((dim * 4 - temp_seg.shape[1]) // 2), ((dim * 4 - temp_seg.shape[1]) // 2)), \
                                     (((dim * 4 - temp_seg.shape[2]) // 2), ((dim * 4 - temp_seg.shape[2]) // 2))),
                          'constant')

        # This line is made to just make it work on the brats dataset CHECK THE DESCRIPTION
        temp_seg[temp_seg == 4] = 3
        temp_seg = tensorflow.keras.utils.to_categorical(temp_seg, num_classes=int(classes))
        mode1 = []
        mode2 = []
        mode3 = []
        mode4 = []
        # The previous local variables will be used to store all the plots generated for each mode available
        for mode in all_modes:
            curr = all_modes.index(mode)
            print(type(curr))
            temp_img = nib.load(mode[i]).get_fdata()

            dim = max(temp_img.shape) // 4
            temp_img = np.pad(temp_img, ((((dim * 4 - temp_img.shape[0]) // 2), ((dim * 4 - temp_img.shape[0]) // 2)), \
                                         (((dim * 4 - temp_img.shape[1]) // 2), ((dim * 4 - temp_img.shape[1]) // 2)), \
                                         (((dim * 4 - temp_img.shape[2]) // 2), ((dim * 4 - temp_img.shape[2]) // 2))),
                              'constant')

            mode_type = mode[i].split("_")[-1].split(".")[0]
            image = scaler.fit_transform(temp_img.reshape(-1, temp_img.shape[-1])).reshape(temp_img.shape)

            labeled_image = np.zeros_like(temp_seg[:, :, :, 1:])
            # We add a fourth dimention depending on the number of classes
            # We also empty out the tumor and add in the values in the orray based on the one hot encoding
            for c in range(int(classes) - 1):
                labeled_image[:, :, :, c] = image * (temp_seg[:, :, :, 0])
                labeled_image += temp_seg[:, :, :, 1:]

            # Fetch local variables
            variables = locals()
            for s in range(temp_img.shape[0] - 1):
                # Depending on the number of modes we generate the list of plots that will be used as frames
                if len(all_modes) != 1:
                    im1 = axs[curr, 0].imshow(rot(labeled_image[:, :, s, :], angle=270), animated=True)
                    im2 = axs[curr, 1].imshow(rot(labeled_image[:, s, :, :], angle=90), animated=True)
                    im3 = axs[curr, 2].imshow(rot(labeled_image[s, :, :, :], angle=90), animated=True)
                else:
                    # In case only one mode was available since a fig of 1 x 3 is treated as a vector (1D)
                    im1 = axs[0].imshow(rot(labeled_image[:, :, s, :], angle=270), animated=True)
                    im2 = axs[1].imshow(rot(labeled_image[:, s, :, :], angle=90), animated=True)
                    im3 = axs[2].imshow(rot(labeled_image[s, :, :, :], angle=90), animated=True)

                variables["mode" + str(curr + 1)].append([im1, im2, im3])

        # Get the first three slices of each view for each modality to display in the same frame to avoid flickering
        for w in range(np.shape(mode1)[0]):
            # Here we check what modes are available and creat a Frame where all the first graph objects can be
            # Displayed
            if len(all_modes) == 1:
                gif.append([mode1[w][0], mode1[w][1], mode1[w][2]])
            if len(all_modes) == 2:
                gif.append([mode1[w][0], mode1[w][1], mode1[w][2], mode2[w][0], mode2[w][1], mode2[w][2]])
            if len(all_modes) == 3:
                gif.append([mode1[w][0], mode1[w][1], mode1[w][2], mode2[w][0], mode2[w][1], mode2[w][2], \
                            mode3[w][0], mode3[w][1], mode3[w][2]])
            if len(all_modes) == 4:
                gif.append([mode1[w][0], mode1[w][1], mode1[w][2], mode2[w][0], mode2[w][1], mode2[w][2], \
                            mode3[w][0], mode3[w][1], mode3[w][2], mode4[w][0], mode4[w][1], mode4[w][2]])

        print("Creating gif for patient " + str(i + 1) + "....")
        ani2 = animate.ArtistAnimation(fig, gif, interval=60, blit=False, repeat_delay=250)

        # Directory to be saved in
        ani2.save("C:\\Users\\bedox\\Desktop\\Data gif generated\\Patient_no_" + str(i + 1) + ".gif")
        gif.clear()


if __name__ == '__main__':
    gif_gen()
    # this allows the function to be run on import outisde main module
    # filename.function()
