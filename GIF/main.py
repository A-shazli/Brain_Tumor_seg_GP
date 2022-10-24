from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from imutils import rotate as rot
from gui import Ui_MainWindow
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
import matplotlib.animation as animate
matplotlib.use('Qt5Agg')
#matplotlib.pyplot.subplots_adjust(left=0.1, bottom=None, right=0.2, top=None, wspace=1, hspace=1)

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
        self.images_seg = []
        print("here1")
        for i in range(self.input_image_data.shape[2]): #gets the number of slices to iterate

            im_sag = self.ui.axes1.imshow(rot(self.input_image_data[i, :, :], angle=90), animated=True, cmap=plt.cm.gray)
            im_sag_seg = self.ui.axes1.imshow(rot(self.input_seg_data[i, :, :], angle=90), animated=True,
                                          cmap=plt.cm.gray)
            self.images.append([im_sag])
            self.images_seg.append([im_sag_seg])


            im_cor = self.ui.axes2.imshow(rot(self.input_image_data[:, i, :], angle=90) , animated=True, cmap=plt.cm.gray)
            im_cor_seg = self.ui.axes2.imshow(rot(self.input_seg_data[:, i, :], angle=90), animated=True,
                                              cmap=plt.cm.gray)
            self.images.append([im_cor])
            self.images_seg.append([im_cor_seg])
            #
            #
            im_ax = self.ui.axes3.imshow(rot(self.input_image_data[:, :, i], angle=90), animated=True, cmap=plt.cm.gray)
            im_ax_seg = self.ui.axes3.imshow(rot(self.input_seg_data[:, :, i], angle=90), animated=True, cmap=plt.cm.gray)
            self.images.append([im_ax])
            self.images_seg.append([im_ax_seg])

        self.ani = animate.ArtistAnimation(self.ui.figure1, self.images, interval=5, \
                                           blit=True, repeat_delay=500)
        self.ani1 = animate.ArtistAnimation(self.ui.figure1, self.images_seg, interval=5, \
                                           blit=True, repeat_delay=500)

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