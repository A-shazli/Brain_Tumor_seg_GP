from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from imutils import rotate as rot
from gui import Ui_MainWindow
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
import imageio
import numpy as np
import matplotlib.animation as animate
matplotlib.use('Qt5Agg')
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\bedox\Desktop\ffmpeg-2022-10-27-git-00b03331a0-essentials_build\bin\ffmpeg.exe'
import ffmpeg
class Logic(QtWidgets.QMainWindow):
    def __init__(self):
        super(Logic, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.Open.triggered.connect(self.load)
        self.ui.AxialSlider.valueChanged.connect(self.display_slice)
        self.ui.CoronalSlider.valueChanged.connect(self.display_slice)
        self.ui.SagittalSlider.valueChanged.connect(self.display_slice)
        self.ui.opacity_slider.valueChanged.connect(self.create_gif)

    def load(self):
        path,format  = QtWidgets.QFileDialog.getOpenFileName(None, "Load Data", "")
        format = path.split('.')

        if path == "":
           pass
        else:
            if format[1] != "nii":
                self.show_popup("Not a nifti file", 'Please upload a compatible file')
            else:
                input_image = nib.load(path) # access data as numpy array to be able to plot
                input_seg = nib.load("C:\\Users\\bedox\\Desktop\\DATA SETS\\Brats2021\\BraTS2021_00495_seg.nii.gz")
                self.input_seg_data = input_seg.get_fdata()
                self.input_image_data = input_image.get_fdata()
                print(input_image.ndim)
                leN = self.input_image_data.shape[2] -1 #to avoid going out of bounds
                self.ui.AxialSlider.setMaximum(leN)
                self.ui.SagittalSlider.setMaximum(leN)
                self.ui.CoronalSlider.setMaximum(leN)
                self.create_gif()
                self.display_slice()

    def create_gif(self):

        self.images = []

        print("here1")
        for i in range(self.input_image_data.shape[2]): #gets the number of slices to iterate

            im_sag = self.ui.axes1.imshow(rot(self.concate(self.input_image_data[i, :, :], self.input_seg_data[i, :, :], self.ui.opacity_slider.value()), angle=90), animated=True, cmap=plt.cm.gray)
            self.images.append([im_sag])


            im_cor = self.ui.axes2.imshow(rot(self.concate(self.input_image_data[:, i, :], self.input_seg_data[:, i, :]), angle=90) , animated=True, cmap=plt.cm.gray, alpha=0.6)
            self.images.append([im_cor])

            im_ax = self.ui.axes3.imshow(rot(self.concate(self.input_image_data[:, :, i], self.input_seg_data[:, :, i]), angle=90), animated=True, cmap=plt.cm.gray, alpha=0.6)
            self.images.append([im_ax])





        self.ani = animate.ArtistAnimation(self.ui.figure1, self.images, interval=12, \
                                           blit=True, repeat_delay=500)

        # writergif = animate.FFMpegWriter(fps=50)
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

        mult = image * (seg * opacity)
        masked = image + mult

        return masked


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


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    style = "style.stylesheet"
    fh = open(style).read()
    app.setStyleSheet(fh)
    logic = Logic()
    logic.show()
    sys.exit(app.exec_())