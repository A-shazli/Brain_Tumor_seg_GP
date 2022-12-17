from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from imutils import rotate as rot
import tensorflow
from gui import Ui_MainWindow
import matplotlib
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.animation as animate
matplotlib.use('Qt5Agg')
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\bedox\Desktop\ffmpeg-2022-10-27-git-00b03331a0-essentials_build\bin\ffmpeg.exe'
from nilearn.image import resample_img

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
        path, format = QtWidgets.QFileDialog.getOpenFileName(None, "Load Data", "")
        self.name = path.split("/")[-1].split(".")[0]
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
                print(np.shape(input_image.get_fdata()))
                input_image = resample_img(input_image, target_affine=np.eye(3), interpolation='nearest')
                input_seg = resample_img(input_seg, target_affine=np.eye(3),
                                           interpolation='nearest')
                self.input_seg_data = input_seg.get_fdata()
                input_image_data = input_image.get_fdata()
                self.input_image_data = self.scaler.fit_transform(input_image_data
                    .reshape(-1, input_image_data.shape[-1])).reshape(input_image_data.shape)

                print( self.input_image_data.max())
                axial = self.input_image_data.shape[2] -1 #to avoid going out of bounds
                coronal = self.input_image_data.shape[1] - 1
                sagital = self.input_image_data.shape[0] - 1
                self.ui.AxialSlider.setMaximum(axial)
                self.ui.SagittalSlider.setMaximum(sagital)
                self.ui.CoronalSlider.setMaximum(coronal)
                self.create_gif()
                self.display_slice()

    def create_gif(self):
        print("hi")
        # the labels on some datasets are not integers
        self.input_seg_data = np.around(self.input_seg_data)
        self.input_seg_data[self.input_seg_data == 4] = 3
        self.input_seg_data = tensorflow.keras.utils.to_categorical(self.input_seg_data, num_classes=4)
        print("hiiii")
        self.labeled_image = np.zeros_like(self.input_seg_data[:, :, :, 1:])
        self.labeled_image[:, :, :, 0] = self.input_image_data * (self.input_seg_data[:, :, :, 0])
        self.labeled_image[:, :, :, 1] = self.input_image_data * (self.input_seg_data[:, :, :, 0])
        self.labeled_image[:, :, :, 2] = self.input_image_data * (self.input_seg_data[:, :, :, 0])
        self.labeled_image += self.input_seg_data[:, :, :, 1:]
        print("hi2")
        self.images = []

        print("here1")
        for i in range(self.input_image_data.shape[2]): #gets the number of slices to iterate

            im_ax = self.ui.axes1.imshow(rot(self.labeled_image[:, :, i, :], angle=90), animated=True)

            self.images.append([im_ax])


        self.ani = animate.ArtistAnimation(self.ui.figure1, self.images, interval=50, \
                                           blit=True, repeat_delay=500)

        self.ani.save("C:\\Users\\bedox\\Desktop\\Data gif generated\\" + self.name + "_axial_view.gif" )
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
        self.ui.axes.imshow(rot(self.labeled_image[:, :, self.ui.AxialSlider.value(), :], angle=90))
        self.ui.canvas3.draw()

        self.ui.axes_cor.clear()
        self.ui.axes_cor.axis("off")
        self.ui.axes_cor.imshow(rot(self.labeled_image[:, self.ui.CoronalSlider.value(), :, :], angle=90))
        self.ui.canvas1.draw()

        self.ui.axes_sag.clear()
        self.ui.axes_sag.axis("off")
        self.ui.axes_sag.imshow(rot(self.labeled_image[self.ui.SagittalSlider.value(), :, :, :], angle=90))
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