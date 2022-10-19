from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

from gui import Ui_MainWindow
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
import numpy as np
import matplotlib.animation as animate
matplotlib.use('Qt5Agg')
from PyQt5.QtGui import QMovie
class Logic(QtWidgets.QMainWindow):
    def __init__(self):
        super(Logic, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.Open.triggered.connect(self.load)
        self.ui.AxialSlider.valueChanged.connect(self.display_slice)
        self.ui.CoronalSlider.valueChanged.connect(self.display_slice)
        self.ui.SagittalSlider.valueChanged.connect(self.display_slice)


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

            im_sag = self.ui.axes1.imshow(self.input_image_data[i, :, :], animated=True)
            self.images.append([im_sag])


            im_cor = self.ui.axes2.imshow(self.input_image_data[:, i, :] , animated=True)
            self.images.append([im_cor])
            #
            #
            im_ax = self.ui.axes3.imshow(self.input_image_data[:, :, i], animated=True)
            self.images.append([im_ax])

        self.ani = animate.ArtistAnimation(self.ui.figure1, self.images, interval=25, \
                                           blit=True, repeat_delay=500)
        print("here2")
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


    def display_slice(self):

        self.ui.axes.clear()
        self.ui.axes.axis("off")
        self.ui.axes.imshow(self.input_image_data[:, :, self.ui.AxialSlider.value()], cmap=plt.cm.gray)
        self.ui.canvas3.draw()

        self.ui.axes_cor.clear()
        self.ui.axes_cor.axis("off")
        self.ui.axes_cor.imshow(self.input_image_data[:, self.ui.CoronalSlider.value(), :], cmap=plt.cm.gray)
        self.ui.canvas1.draw()

        self.ui.axes_sag.clear()
        self.ui.axes_sag.axis("off")
        self.ui.axes_sag.imshow(self.input_image_data[self.ui.SagittalSlider.value(), :, :], cmap=plt.cm.gray)
        self.ui.canvas2.draw()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    logic = Logic()
    logic.show()
    sys.exit(app.exec_())