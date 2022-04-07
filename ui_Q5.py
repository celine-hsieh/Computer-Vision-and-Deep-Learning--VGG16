import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QLabel, QWidget, QLineEdit
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QPushButton, QGroupBox, QMessageBox
from PyQt5.QtCore import Qt, QMetaObject

import cv2 as cv
import glob
import os
import numpy as np
import utils_Q5
__appname__ = "2021 Opencvdl Hw1 Question 5"

class windowUI(object):
    """
    Set up UI
    please don't edit
    """
    def setupUI(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640,480)
        MainWindow.setWindowTitle(__appname__)


        # 5. VGG Test Group
        VGG_Group = QGroupBox("5. VGG Test")
        group_V1_vBoxLayout = QVBoxLayout(VGG_Group)

        self.button5_1 = QPushButton("5.1 Show Train Images")
        self.button5_2 = QPushButton("5.2 Show HyperParameter")
        self.button5_3 = QPushButton("5.3 Show Model Shortcut")
        self.button5_4 = QPushButton("5.4 Show Accuracy")

        Test_Group = QGroupBox("5.5 Test")
        group_V1_5_vBoxLayout = QVBoxLayout(Test_Group)

        select_layout, self.edit_5_5 = self.edit_Text("Select image: ")
        group_V1_5_vBoxLayout.addLayout(select_layout)
        self.button5_5 = QPushButton("5.5 Test")
        group_V1_5_vBoxLayout.addWidget(self.button5_5)
        
        self.button5_6 = QPushButton("5.6 Train")

        group_V1_vBoxLayout.addWidget(self.button5_1)
        group_V1_vBoxLayout.addWidget(self.button5_2)
        group_V1_vBoxLayout.addWidget(self.button5_3)
        group_V1_vBoxLayout.addWidget(self.button5_4)
        
        group_V1_vBoxLayout.addWidget(Test_Group)
        group_V1_vBoxLayout.addWidget(self.button5_6)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        vLayout = QHBoxLayout()
        vLayout.addWidget(VGG_Group)
        self.centralwidget.setLayout(vLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        QMetaObject.connectSlotsByName(MainWindow)

    @staticmethod
    def edit_Text(title:str, unit = "", showUnit= False):
        hLayout = QHBoxLayout()

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_label.setFixedWidth(60)
        unit_label = QLabel(unit)
        unit_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        unit_label.setFixedWidth(30)
        editText = QLineEdit("1")
        editText.setFixedWidth(50)
        editText.setAlignment(Qt.AlignRight)
        editText.setValidator(QIntValidator())

        hLayout.addWidget(title_label, alignment=Qt.AlignLeft)
        hLayout.addWidget(editText)
        if showUnit:
            hLayout.addWidget(unit_label)
        return hLayout, editText

class MainWindow(QMainWindow, windowUI):

    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUI(self)
        self.initialValue()
        self.buildUi()

    def buildUi(self):
        self.button5_1.clicked.connect(self.show_sample_image)
        self.button5_2.clicked.connect(self.show_hyper)
        self.button5_3.clicked.connect(self.show_model)
        self.button5_4.clicked.connect(self.show_acc)
        self.button5_5.clicked.connect(self.test)
        self.button5_6.clicked.connect(self.train)  
    
    def initialValue(self):
        self.Q5_model = utils_Q5.Q5_Cifar10(modeTrain=False)
        # self.model.load_test_dataset(show_sample=True)
        self.load = False
        self.run_test = False
        self.button5_6.setEnabled(self.Q5_model.modeTrain)
        pass

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))
    
    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def show_sample_image(self, show_sample = True):
        print("Please wait a second")
        self.status("Load CiFar10 Dataset, Please wait a second")
        self.setEnabled(False)
        self.testset = self.Q5_model.load_test_dataset(show_sample=show_sample)
        self.setEnabled(True)
        self.load = True
        pass

    def show_hyper(self):
        self.status("Showing hyperparameters in command line")
        self.Q5_model.show_hyperparameter()
        pass

    def show_model(self):
        self.status("Showing model summary in command line")
        self.Q5_model.show_model()
        pass

    def show_acc(self):
        path = "model/history.json"
        if os.path.exists(path):
            history = utils_Q5.load_history(path)
            utils_Q5.show_acc_plot(history)
        else:
            self.errorMessage(u'Error opening file',
                                u"<p>Make sure <i></i> is a json file.")
            self.status("Error reading")
        pass

    def test(self):
        if not self.load:
            self.show_sample_image(show_sample=False)
        index = int(self.edit_5_5.text())
        self.setEnabled(False)
        self.Q5_model.test(index)
        self.setEnabled(True)
        pass

    def train(self):
        
        self.Q5_model.load_train_dataset(1)
        self.Q5_model.train()
        pass




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(500, 150, 500, 300)
    window.show()
    sys.exit(app.exec_())