from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMessageBox
from imutils import rotate as rot
from matplotlib.figure import Figure

from gui import Ui_MainWindow
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import gc
import os
import numpy as np
import glob
import matplotlib.animation as animate
matplotlib.use('Qt5Agg')
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\bedox\Desktop\ffmpeg-2022-10-27-git-00b03331a0-essentials_build\bin\ffmpeg.exe'

class Logic(QtWidgets.QMainWindow):
    def __init__(self):
        super(Logic, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


        self.ui.Open.triggered.connect(self.load)
        self.ui.AxialSlider.valueChanged.connect(self.display_slice)
        self.ui.CoronalSlider.valueChanged.connect(self.display_slice)
        self.ui.SagittalSlider.valueChanged.connect(self.display_slice)
        self.ui.opacity_slider.sliderPressed.connect(self.pause_ani)
        self.ui.opacity_slider.sliderReleased.connect(self.create_gif)

    def load(self):
        path, format = QtWidgets.QFileDialog.getOpenFileName(None, "Load Data", "")
        format = path.split('.')
        if path == "":
           pass
        else:
            if format[1] != "nii":
                self.show_popup("Not a nifti file", 'Please upload a compatible file')
            else:
                self.scaler = MinMaxScaler()
                input_image = nib.load(path) # access data as numpy array to be able to plot
                path_seg, form = QtWidgets.QFileDialog.getOpenFileName(None, "Load SEGMENT Data", "")
                input_seg = nib.load(path_seg)
                self.input_seg_data = input_seg.get_fdata()
                input_image_data = input_image.get_fdata()
                self.input_image_data = self.scaler.fit_transform(input_image_data.reshape(-1, input_image_data.shape[-1])).reshape(input_image_data.shape)

                print( self.input_image_data.max())
                leN = self.input_image_data.shape[2] -1 #to avoid going out of bounds
                self.ui.AxialSlider.setMaximum(leN)
                self.ui.SagittalSlider.setMaximum(leN)
                self.ui.CoronalSlider.setMaximum(leN)
                self.create_gif()
                self.display_slice()
                self.gif_img_mask(path)

    def create_gif(self):



        self.images = []

        print("here1")
        for i in range(self.input_image_data.shape[2]): #gets the number of slices to iterate

            im_sag = self.ui.axes1.imshow(rot(self.concate(self.input_image_data[i, :, :], self.input_seg_data[i, :, :]), angle=90), animated=True, cmap=plt.cm.gray)
            self.images.append([im_sag])

            im_cor = self.ui.axes2.imshow(rot(self.concate(self.input_image_data[:, i, :], self.input_seg_data[:, i, :]), angle=90) , animated=True, cmap=plt.cm.gray)
            self.images.append([im_cor])

            im_ax = self.ui.axes3.imshow(rot(self.concate(self.input_image_data[:, :, i], self.input_seg_data[:, :, i]), angle=90), animated=True, cmap=plt.cm.gray)
            self.images.append([im_ax])






        self.ani = animate.ArtistAnimation(self.ui.figure1, self.images, interval=12, \
                                           blit=True, repeat_delay=500)

        # writergif = animate.FFMpegWriter(fps=60)
        # self.ani.save("ww.gif", writer=writergif)

        self.ui.canvas.draw()

    def show_popup(self, message, information):
        msg = QMessageBox()
        msg.setWindowTitle("Message")
        msg.setText(message)
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Ok)
        msg.setInformativeText(information)
        msg.exec_()


    def concate(self, image, seg):


        # mult =  image * (seg)
        #
        # masked = image + mult
        # mult = cv2.bitwise_or(image, seg)
        return image + seg * self.ui.opacity_slider.value()/10


    def concate_2(self, image, seg):


        # mult =  image * (seg)
        #
        # masked = image + mult
        # mult = cv2.bitwise_or(image, seg)
        return image + seg

    def pause_ani(self):
        del self.images
        del self.ani
        gc.collect()
        self.ui.figure1.clear()


    def display_slice(self):

        self.ui.axes.clear()
        self.ui.axes.axis("off")
        self.ui.axes.imshow(rot(self.input_image_data[:, :, self.ui.AxialSlider.value()],angle=90), cmap=plt.cm.gray)
        self.ui.canvas3.draw()

        self.ui.axes_cor.clear()
        self.ui.axes_cor.axis("off")
        self.ui.axes_cor.imshow(rot(self.input_image_data[:, self.ui.CoronalSlider.value(), :],angle=90), cmap=plt.cm.gray)
        self.ui.canvas1.draw()

        self.ui.axes_sag.clear()
        self.ui.axes_sag.axis("off")
        self.ui.axes_sag.imshow(rot(self.input_image_data[self.ui.SagittalSlider.value(), :, :],angle=90), cmap=plt.cm.gray)
        self.ui.canvas2.draw()




    def gif_img_mask(self, path):
        pa = ""
        gif = []
        path = path.split("/")
        print(path)
        for p in range(len(path) -2):
            pa += path[p] + "/"
        pa = pa[:-1]
        print(pa)
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
        print(len(t2_list))

        for img in range(len(t2_list)):
            temp_img = nib.load(t2_list[img]).get_fdata()
            temp_img = self.scaler.fit_transform(temp_img.reshape(-1, temp_img.shape[-1])).reshape(temp_img.shape)
            temp_seg = nib.load(mask_list[img]).get_fdata()
            for i in range(temp_img.shape[2]):
                im1 = ax1.imshow(self.concate_2(temp_img[:, :, i], temp_seg[:, :, i]), animated=True, cmap='gray')
                gif.append([im1])
                im2 = ax2.imshow(self.concate_2(temp_img[:, i, :], temp_seg[:, i, :]), animated=True, cmap='gray')
                gif.append([im2])
                im3 = ax3.imshow(self.concate_2(temp_img[i, :, :], temp_seg[i, :, :]), animated=True, cmap='gray')
                gif.append([im3])
            ani2 = animate.ArtistAnimation(fig, gif, interval=30, \
                                          blit=True, repeat_delay=500)
            writergif = animate.FFMpegWriter(fps=30)
            ani2.save("C:\\Users\\bedox\\Desktop\\Data gif generated\\gif_no_" + str(img) + ".mov", writer=writergif)
            gif.clear()
            del temp_img
            del temp_seg
            gc.collect()
        print("DONE")

        # for subdir, dirs, files in os.walk(pa):
        #     for file in files:
        #         print(os.path.join(subdir, file))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    style = "style.stylesheet"
    fh = open(style).read()
    app.setStyleSheet(fh)
    logic = Logic()
    logic.show()
    sys.exit(app.exec_())