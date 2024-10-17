import os, sys, math, time, pdb

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QTableWidgetItem, QMessageBox, QWidget
from PyQt5.QtGui import QImage, QPixmap, QColor, QIcon


from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from GUI import Ui_MainWindow 
import cv2
import numpy as np 
import pandas as pd

# for ecg signal processing
import wfdb 

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


        # adding some stuff 
        self.setWindowTitle("Patient Health data Processing")
        self.setStyleSheet('background-color: #333; color: white;')


        # **** For image processing ****
        self.in_img = 0 # Pyqt img 
        self.out_img = 0 # Pyqt img
        self.image = None # cv2 image 


        self.tmp = 0

        self.k_size = self.ui.spinBox_kernel.value()


        # callback connection 
        self.ui.actionopen.triggered.connect(self.callback_open_image)

        self.ui.btn_loadImage.clicked.connect(self.callback_open_image)
        self.ui.btn_grayscale.clicked.connect(self.callback_grayscale)
        self.ui.btn_blur.clicked.connect(self.callback_gauss)
        self.ui.btn_cany.clicked.connect(self.callback_canny)
        self.ui.btn_laplacian.clicked.connect(self.callback_laplacian)
        self.ui.btn_thresholding.clicked.connect(self.callback_thresholding)
  


        self.ui.btn_loadECG.clicked.connect(self.callback_load_ecg)
        self.ui.btn_FFT.clicked.connect(self.plotFFT)
        self.ui.btn_periodogram.clicked.connect(self.callback_periodogram)


        # **** For ECG spectral analysis ***
        self.ecg_record = None
        self.canvas_in = None
        self.canvas_out = None

        self.toolbar_in = None
        self.toolbar_out = None

        self.dimension = 1
        self.signal= None
        self.fs = None
        self.begin = 0
        self.end = 7440

        self.installEventFilter(self)



    def callback_open_image(self):

        fname, ext = QFileDialog.getOpenFileName(self, 'Open File', '', "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if fname:

            # enable the buttons 
            for button in [self.ui.btn_grayscale ,self.ui.btn_blur, self.ui.btn_cany, self.ui.btn_laplacian, self.ui.btn_thresholding]: 
                button.setEnabled(True)

            self.image = cv2.imread(fname)
            self.tmp = self.image

            # shape = self.image.shape 
            # rgb_image = np.full((shape[0], shape[1], 3), 1, dtype=np.uint8)
            self._display_image(window=1, myimage=self.image)


    def callback_grayscale(self): 
        print('running grayscale')
        if(self.image.shape[-1] != 3 or len(self.image.shape) < 2): 
            print('image already in grayscale! ')
            return 

        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self._display_image(2)

    def callback_gauss(self): 
        print('running gaussian')
        k_size = self.ui.spinBox_kernel.value()
        self.image = cv2.GaussianBlur(self.image, (k_size, k_size),0)


        self._display_image(2) 

    def callback_canny(self): 
        print('running canny')
        # noise reduction
        blurred = cv2.GaussianBlur(self.image, (5, 5), 1.4)

        self.image = cv2.Canny(blurred, threshold1=100, threshold2=200)
        self._display_image(2)

    def callback_laplacian(self): 
        print('running laplacian')

        if (self.image.shape[-1] != 3 or len(self.image.shape) < 2) == False: 
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image


        # Step 1: Noise reduction
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 1.4)

        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

        # Step 3: Get the edges (absolute value)
        self.image = np.uint8(np.absolute(laplacian))


        self._display_image(2) 

    def callback_thresholding(self): 
        print('running thresholding')
        self._display_image(2) 

    

    # callbacks for ecg 

    def plotData(self):

        self.rm_mpl(window=1)

        print('plotting ! ')


        # Prepare the figure and canvas
        fig = Figure()
        self.canvas_in = FigureCanvas(fig)
        self.ui.verticalLayout_7.addWidget(self.canvas_in)

        # Create the plot
        ax1f1 = fig.add_subplot(111)
        ax1f1.set_title('ECG Signal')
        ax1f1.set_xlabel("X-axis", fontsize=14)
        ax1f1.set_ylabel("Y-axis", fontsize=14)



        ax1f1.plot(self.signal[self.begin:self.end, self.dimension], label=f'Channel {self.dimension}')

        # Reformat the coordinates near the toolbar to not show scientific notation
        ax1f1.format_coord = lambda x, y: f"x={x:1.4f}, y={y:1.4f}"


        # Draw everything and add toolbar
        self.canvas_in.draw()
        self.toolbar_in = NavigationToolbar(self.canvas_in, self.ui.mplwindow, coordinates=True)
        self.ui.verticalLayout_7.addWidget(self.toolbar_in)


    def plotFFT(self): 
        self.rm_mpl(window=2)

        print('plotting FFT ! ')
        # Prepare some dummy data

        if self.signal is None or self.signal.size == 0:
            print('there is no read signal ')
            return
        

        N = len(self.signal[self.begin:self.end,self.dimension])

        yf = np.fft.fft(self.signal[self.begin:self.end, self.dimension])
        xf = np.fft.fftfreq(N, 1/self.fs)

        amplitude = np.abs(yf)[:] * (2.0 / N) 

        # Prepare the figure and canvas
        fig = Figure()
        self.canvas_out = FigureCanvas(fig)
        self.ui.verticalLayout_8.addWidget(self.canvas_out)

        # Create the plot
        ax1f1 = fig.add_subplot(111)
        ax1f1.set_title('FFT of the ecg')
        ax1f1.set_xlabel("freqencies", fontsize=14)
        ax1f1.set_ylabel("Amplitude", fontsize=14)


        # amplitude normalized
        amplitude = np.abs(yf)[:] * (2.0 / N)
        ax1f1.plot(xf, amplitude)
        ax1f1.set_xlim(0, self.fs / 2) 

        # Reformat the coordinates near the toolbar to not show scientific notation
        ax1f1.format_coord = lambda x, y: f"x={x:1.4f}, y={y:1.4f}"


        # Draw everything and add toolbar
        self.canvas_out.draw()
        self.toolbar_out = NavigationToolbar(self.canvas_out, self.ui.mplwindow, coordinates=True)
        self.ui.verticalLayout_8.addWidget(self.toolbar_out)



    def plotPeriodogram(self): 
        self.rm_mpl(window=2)

        print('plotting FFT ! ')
        # Prepare some dummy data

        if self.signal is None or self.signal.size == 0:
            print('there is no read signal ')
            return
        

        N = len(self.signal[self.begin:self.end,self.dimension])

        yf = np.fft.fft(self.signal[self.begin:self.end, self.dimension])
        xf = np.fft.fftfreq(N, 1/self.fs)

        amplitude = np.abs(yf)[:] * (2.0 / N) 

        # Prepare the figure and canvas
        fig = Figure()
        self.canvas_out = FigureCanvas(fig)
        self.ui.verticalLayout_8.addWidget(self.canvas_out)

        # Create the plot
        ax1f1 = fig.add_subplot(111)
        ax1f1.set_title('FFT of the ecg')
        ax1f1.set_xlabel("freqencies", fontsize=14)
        ax1f1.set_ylabel("Amplitude", fontsize=14)


        # amplitude normalized
        amplitude = np.abs(yf)[:] * (2.0 / N)
        ax1f1.plot(xf, amplitude)
        ax1f1.set_xlim(0, self.fs / 2) 

        # Reformat the coordinates near the toolbar to not show scientific notation
        ax1f1.format_coord = lambda x, y: f"x={x:1.4f}, y={y:1.4f}"


        # Draw everything and add toolbar
        self.canvas_out.draw()
        self.toolbar_out = NavigationToolbar(self.canvas_out, self.ui.mplwindow, coordinates=True)
        self.ui.verticalLayout_8.addWidget(self.toolbar_out)
 



    def rm_mpl(self, window):

        layout = self.ui.verticalLayout_7 if window==1 else self.ui.verticalLayout_8
        toolbar = self.toolbar_in if window == 1 else self.toolbar_out
        canvas = self.canvas_in if window == 1 else self.canvas_out 

        if(canvas): 
            layout.removeWidget(canvas)
            canvas.close()

        if(toolbar):
            layout.removeWidget(self.toolbar_in)
            self.toolbar_in.close()





    def callback_load_ecg(self):    
        fname, ext = QFileDialog.getOpenFileName(self, 'Open File', '', "Image Files (*.dat)")
        if fname:

            base_path = os.path.splitext(fname)[0]

            try:
                self.ecg_record = wfdb.rdrecord(base_path)
                self.signal = self.ecg_record.p_signal
                self.fs = self.ecg_record.fs
            except Exception as e:
                print(f"Error reading file: {e}")


            # enable the buttons 
            for button in [self.ui.btn_FFT, self.ui.btn_periodogram]:
                button.setEnabled(True)



            self.plotData()


    def callback_FFT(self): 
        pass 

    def callback_periodogram(self):
        pass 

    


    def _display_image(self, window, myimage=np.zeros(1)):

        # If you pass an image as argument and it contains data, replace self.image with this image
        if myimage.any():
            self.image = myimage

        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        img = img.rgbSwapped()

        # Adapt the size of the image based on the size of the label -> static version
        if window == 1:
            self.in_img = img.scaled(self.ui.labelnputImage.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            print('type of self.in_img : ', type(self.in_img))
            self.ui.labelnputImage.setPixmap(QPixmap.fromImage(self.in_img))

        if window == 2:
            self.out_img = img.scaled(self.ui.labelOutputImage.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.labelOutputImage.setPixmap(QPixmap.fromImage(self.out_img))
            self.boutputloaded = True

        if window == 3: 
            pass 


        if window ==4: 
            pass 

    # @override : NOTE: you should keep the name as it is
    def eventFilter(self, obj, event):
        # Adapt the size of the image based on the size of the label -> dynamic version
        if event.type() == QtCore.QEvent.Resize:
            if self.in_img != 0:
                self.ui.labelnputImage.setMinimumSize(1, 1)
                self.ui.labelnputImage.setPixmap(QPixmap.fromImage(self.in_img.scaled(self.ui.labelnputImage.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            if self.out_img != 0:
                self.ui.labelOutputImage.setMinimumSize(1, 1)
                self.ui.labelOutputImage.setPixmap(QPixmap.fromImage(self.out_img.scaled(self.ui.labelOutputImage.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)))

        return super().eventFilter(obj, event)







if __name__ == '__main__':
    app = QApplication(sys.argv)

    # To avoid weird behaviors (smaller items, ...) on big resolution screens
    app.setStyle("fusion")
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())