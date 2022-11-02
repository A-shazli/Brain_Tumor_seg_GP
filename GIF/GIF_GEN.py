import nibabel as nib
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.animation as animate
from imutils import rotate as rot
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\bedox\Desktop\ffmpeg-2022-10-27-git-00b03331a0-essentials_build\bin\ffmpeg.exe'

def concate_2(image, seg):
    # mult =  image * (seg)
    #
    # masked = image + mult
    # mult = cv2.bitwise_or(image, seg)
    return image + seg*0.5



pa = input("Please enter path: ")   #"C:\\Users\\bedox\Desktop\DATA SETS\Brats2021"
t2_list = sorted(glob.glob(pa + '/*/*t2.nii.gz'))
t1ce_list = sorted(glob.glob(pa + '/*/*t1ce.nii.gz'))
flair_list = sorted(glob.glob(pa + '/*/*flair.nii.gz'))
mask_list = sorted(glob.glob(pa + '/*/*seg.nii.gz'))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
plt.style.use('dark_background')
fig.patch.set_facecolor('black')

gif = []
scaler = MinMaxScaler()
for img in range(len(t2_list)):
    temp_img = nib.load(t2_list[img]).get_fdata()
    temp_img = scaler.fit_transform(temp_img.reshape(-1, temp_img.shape[-1])).reshape(temp_img.shape)
    temp_seg = nib.load(mask_list[img]).get_fdata()
    for i in range(temp_img.shape[2]):
        im1 = ax1.imshow(rot(concate_2(temp_img[:, :, i], temp_seg[:, :, i]), angle=90),  animated=True, cmap='gray' )
        gif.append([im1])
        im2 = ax2.imshow(rot(concate_2(temp_img[:, i, :], temp_seg[:, i, :]), angle=90), animated=True, cmap='gray')
        gif.append([im2])
        im3 = ax3.imshow(rot(concate_2(temp_img[i, :, :], temp_seg[i, :, :]), angle=90), animated=True, cmap='gray')
        gif.append([im3])
    ani2 = animate.ArtistAnimation(fig, gif, interval=15, \
                                  blit=True, repeat_delay=500)
    writermov = animate.FFMpegWriter(fps=60)
    ani2.save("C:\\Users\\bedox\\Desktop\\Data gif generated\\gif_no_" + str(img) + ".mov", writer=writermov)
    ax3.clear()
    ax1.clear()
    ax2.clear()
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    gif.clear()
print("DONE")


