import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QLabel, QWidget, QLineEdit
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QPushButton, QGroupBox
from PyQt5.QtCore import Qt, QMetaObject
#from mainwindow.MainWindow import resize

import cv2 as cv
import glob
import os
import numpy as np
import utils
__appname__ = "cvdl Hw1_N76101012"

class windowUI(object):
    
    # Set up UI
    
    def UI(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(800, 300)
        MainWindow.setWindowTitle(__appname__)


        # 1. Calibration Group
        Calibration_Group = QGroupBox("1. Calibration")
        group_V1_vBoxLayout = QVBoxLayout(Calibration_Group)

        self.button1_1 = QPushButton("1.1 Find Corners")
        self.button1_2 = QPushButton("1.2 Find Intrinsic")

        Find_Extrinsic_Group = QGroupBox("1.3 Find Extrinsic")
        group_V1_3_vBoxLayout = QVBoxLayout(Find_Extrinsic_Group)

        select_layout, self.edit_1_3 = self.edit_Text("Select image: ")
        group_V1_3_vBoxLayout.addLayout(select_layout)
        self.button1_3 = QPushButton("1.3 Find Extrinsic")
        group_V1_3_vBoxLayout.addWidget(self.button1_3)
        
        self.button1_4 = QPushButton("1.4 Find Distortion")
        self.button1_5 = QPushButton("1.5 Show Result")

        group_V1_vBoxLayout.addWidget(self.button1_1)
        group_V1_vBoxLayout.addWidget(self.button1_2)
        group_V1_vBoxLayout.addWidget(Find_Extrinsic_Group)
        group_V1_vBoxLayout.addWidget(self.button1_4)
        group_V1_vBoxLayout.addWidget(self.button1_5)

        # 2. Augmented Reality Group
        Augmented_Reality_Group = QGroupBox("2. Augmented Reality")
        group_V2_vBoxLayout = QVBoxLayout(Augmented_Reality_Group)

        self.edit_2 = QLineEdit("OPENCV")
        self.button2_1 = QPushButton("2.1 Show Words on Board")
        self.button2_2 = QPushButton("2.2 Show Words Vertically")
        group_V2_vBoxLayout.addWidget(self.edit_2)
        group_V2_vBoxLayout.addWidget(self.button2_1)
        group_V2_vBoxLayout.addWidget(self.button2_2)
        
        
        # 3. Stereo Disparity Map Group
        Stereo_Disparity_Group = QGroupBox("3. Stereo Disparity Map")
        group_V3_vBoxLayout = QVBoxLayout(Stereo_Disparity_Group)

        self.button3_1 = QPushButton("3.1 Stereo Disparity Map")
        self.button3_2 = QPushButton("3.2 Checking the Disparity Value")
        group_V3_vBoxLayout.addWidget(self.button3_1)
        group_V3_vBoxLayout.addWidget(self.button3_2)

        # 4. Transformation Group
        SIFT_Group = QGroupBox("4. SIFT")
        group_V4_vBoxLayout = QVBoxLayout(SIFT_Group)

        self.button4_1 = QPushButton("4.1 Keypoints")
        self.button4_2 = QPushButton("4.2 Matched keypoints")
        self.button4_3 = QPushButton("4.3 Warp Image")
        group_V4_vBoxLayout.addWidget(self.button4_1)
        group_V4_vBoxLayout.addWidget(self.button4_2)   
        group_V4_vBoxLayout.addWidget(self.button4_3)     

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        vLayout = QHBoxLayout()
        vLayout.addWidget(Calibration_Group)
        vLayout.addWidget(Augmented_Reality_Group)
        vLayout.addWidget(Stereo_Disparity_Group)
        vLayout.addWidget(SIFT_Group)
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
        self.UI(self)
        self.initialValue()
        self.buildUi()
        #self.setFixedSize(411, 247)

    def buildUi(self):
        self.button1_1.clicked.connect(self.find_corners)
        self.button1_2.clicked.connect(self.find_intrinsic)
        self.button1_3.clicked.connect(self.find_extrinsic)
        self.button1_4.clicked.connect(self.find_distortion)
        self.button1_5.clicked.connect(self.show_result)

        self.button2_1.clicked.connect(self.word_on_board)
        self.button2_2.clicked.connect(self.word_Vertical)

        self.button3_1.clicked.connect(self.stereoDisparity)
        self.button3_2.clicked.connect(self.stereoDisparity_value)

        self.button4_1.clicked.connect(self.keypoints_Q4)
        self.button4_2.clicked.connect(self.matchKeypoints)   
        self.button4_3.clicked.connect(self.warpImage)     
    
    def initialValue(self):
        self.images_Q1 = []
        self.images_Q2 = []
        self.images_Q4 = []
        self.q1_1 = False
        self.q1_2 = False
        self.q2_calib = False

        self.char_in_board = [ # coordinate for 6 charter in board (x, y) ==> (w, h)
            [7,5,0], # slot 1
            [4,5,0], # slot 2
            [1,5,0], # slot 3
            [7,2,0], # slot 4
            [4,2,0], # slot 5
            [1,2,0]  # slot 6
        ]
        self.q3_1 = False

        self.q4_1 = False
        self.q4_2 = False

    def find_corners(self):
        width = 11
        height = 8
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points
        objp = np.zeros((height*width, 3), np.float32)
        objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

        # Array to store object points and image points from all the image.
        self.objpoints = [] #3d point in real world space
        self.imgpoints = [] #2d points in image plane
    
        self.images_Q1 = utils.readImages("Dataset/Q1_Image")
        self.setEnabled(False)
        for image in self.images_Q1:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (width,height), None)

            # If found, add object points, image points
            if ret:
                self.objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
                self.imgpoints.append(corners)

                # Draw and Display the corners
                find_corner_image = cv.drawChessboardCorners(image.copy(), (width,height), corners2, ret)
                QApplication.processEvents()

                cv.namedWindow("Find corners", cv.WINDOW_GUI_EXPANDED)
                cv.imshow("Find corners", find_corner_image)
                cv.waitKey(500)
        cv.destroyAllWindows()
        self.setEnabled(True)
        self.q1_1 = True

    def find_intrinsic(self):
        if not self.q1_1:
            self.find_corners()
        h, w = self.images_Q1[0].shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, (w,h), None, None)
        QApplication.processEvents()
        print("Intrinsic matrix: \n", mtx)
        self.dist = dist
        self.rotate_v = rvecs
        self.translate_v = tvecs
        self.mtx = mtx
        self.q1_2 = True

    def find_extrinsic(self):
        if not self.q1_2:
            self.find_intrinsic()
        number_image = int(self.edit_1_3.text())
        if not ((number_image -1) < 0 and (number_image -1)> len(self.images_Q1)):
            rvec = self.rotate_v[number_image - 1]
            tvec = self.translate_v[number_image-1]
            tvec = tvec.reshape(3,1)
            if rvec is not None and tvec is not None:
                Rotation_matrix = cv.Rodrigues(rvec)[0]
                Extrinsic_matrix = np.hstack([Rotation_matrix, tvec])
                print("Extrinsix: \n", Extrinsic_matrix)
        else:
            print("Input error: Please input from 1-15")
        
    def find_distortion(self):
        if not self.q1_2:
            self.find_intrinsic()
        print("Distortion: \n", self.dist[-1])

    def show_result(self):
        if not self.q1_2:
            self.find_intrinsic()
        self.setEnabled(False)
        for image in self.images_Q1:
            h, w = image.shape[:2]
            newcameramatrix, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w,h))
            dst = cv.undistort(image, self.mtx, self.dist, None, newcameramatrix)
            x, y, w, h = roi

            dst = dst[y:y+h, x:x+w]

            print(dst.shape)
            dst = cv.resize(dst, (image.shape[1], image.shape[0]))
            new = utils.concat_image(image, dst)
            QApplication.processEvents()
            cv.namedWindow("Show Result", cv.WINDOW_GUI_EXPANDED)
            cv.imshow("Show Result", new)
            cv.waitKey(500)
        self.setEnabled(True)
        cv.destroyAllWindows()
        pass

    def word_on_board(self):
        if len(self.images_Q2) == 0:
            self.images_Q2 = utils.readImages("Dataset/Q2_Image")
        fs = cv.FileStorage("Dataset/Q2_Image/Q2_Lib/alphabet_lib_onboard.txt", cv.FILE_STORAGE_READ)
        string = self.edit_2.text()[:6]
        if not string.isupper():
            string = string.upper()
        if not self.q2_calib:
            print("Calibrating>>>")
            self.q2_objps, self.q2_imageps = utils.calibration(self.images_Q2)
            QApplication.processEvents()
            self.q2_calib = True
        self.setEnabled(False)
        for index, image in enumerate(self.images_Q2):
            h, w = image.shape[:2]
            draw_image = image.copy()
            ret, intrinsic_mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.q2_objps, self.q2_imageps, (w,h), None, None)
            QApplication.processEvents()
            if ret:
                rvec = np.array(rvecs[index])
                tvec = np.array(tvecs[index]).reshape(3,1)
                for i_char, character in enumerate(string):
                    ch = np.float32(fs.getNode(character).mat())
                    line_list = []
                    for eachline in ch:
                        ach = np.float32([self.char_in_board[i_char], self.char_in_board[i_char]])
                        eachline = np.add(eachline, ach)
                        image_points, jac = cv.projectPoints(eachline, rvec, tvec, intrinsic_mtx, dist)
                        line_list.append(image_points)
                    draw_image = utils.draw_char(draw_image, line_list)
                QApplication.processEvents()
                cv.namedWindow("Word On Board", cv.WINDOW_GUI_EXPANDED)
                cv.imshow("Word On Board", draw_image)
                cv.waitKey(800)
        self.setEnabled(True)   
        pass

    def word_Vertical(self):
        if len(self.images_Q2) == 0:
            self.images_Q2 = utils.readImages("Dataset/Q2_Image")
        fs = cv.FileStorage("Dataset/Q2_Image/Q2_Lib/alphabet_lib_vertical.txt", cv.FILE_STORAGE_READ)
        string = self.edit_2.text()[:6]
        if not string.isupper():
            string = string.upper()
        if not self.q2_calib:
            print("Calibrating>>>")
            self.q2_objps, self.q2_imageps = utils.calibration(self.images_Q2)
            QApplication.processEvents()
            self.q2_calib = True
        self.setEnabled(False)
        for index, image in enumerate(self.images_Q2):
            h, w = image.shape[:2]
            draw_image = image.copy()
            ret, intrinsic_mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.q2_objps, self.q2_imageps, (w,h), None, None)
            QApplication.processEvents()
            if ret:
                rvec = np.array(rvecs[index])
                tvec = np.array(tvecs[index]).reshape(3,1)
                for i_char, character in enumerate(string):
                    ch = np.float32(fs.getNode(character).mat())
                    line_list = []
                    for eachline in ch:
                        ach = np.float32([self.char_in_board[i_char], self.char_in_board[i_char]])
                        eachline = np.add(eachline, ach)
                        image_points, jac = cv.projectPoints(eachline, rvec, tvec, intrinsic_mtx, dist)
                        line_list.append(image_points)
                    draw_image = utils.draw_char(draw_image, line_list)
                QApplication.processEvents()
                cv.namedWindow("Word Vertical", cv.WINDOW_GUI_EXPANDED)
                cv.imshow("Word Vertical", draw_image)
                cv.waitKey(800)   
        self.setEnabled(True)
        pass

    def stereoDisparity(self):
        self.setEnabled(False)
        self.imL = cv.imread("Dataset/Q3_Image/imL.png")
        self.imR = cv.imread("Dataset/Q3_Image/imR.png")

        grayL = cv.cvtColor(self.imL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(self.imR, cv.COLOR_BGR2GRAY)

        disparity_f = utils.disparity(grayL, grayR)
        print(disparity_f.shape)
        self.u8 =utils.process_ouput(disparity_f) 
        self.setEnabled(True)
        #show disparity
        cv.namedWindow("Disparity", cv.WINDOW_GUI_EXPANDED)
        # cv.imshow("Disparity", (disparity - min_disp)/ num_disp)
        cv.imshow("Disparity", self.u8)
        cv.waitKey(1000)
        self.q3_1 = True
        pass
    
    def stereoDisparity_value(self):
        if not self.q3_1:
            self.stereoDisparity()
        cv.namedWindow("Checking Disparity", cv.WINDOW_GUI_EXPANDED)
        utils.map_disparity(self.imL, self.imR, self.u8, "Checking Disparity")
        cv.waitKey(0)

    def keypoints_Q4(self):
        if len(self.images_Q4) == 0:
            self.images_Q4 = utils.readImages("Dataset/Q4_Image", "*.jpg")
        detector, self.matcher = utils.init_feature("brisk")

        self.keypoints = []
        self.descriptors = []
        self.setEnabled(False)
        for image in self.images_Q4:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # keypoint, descriptor = sift.detectAndCompute(gray, None)
            keypoint, descriptor = detector.detectAndCompute(gray, None)
            # keypoint = sift.detect(gray, None)
            self.keypoints.append(keypoint)
            self.descriptors.append(descriptor)
            image_sift = cv.drawKeypoints(image.copy(), keypoint, image.copy())

            cv.namedWindow("Key Point", cv.WINDOW_GUI_EXPANDED)
            cv.imshow("Key Point", image_sift)
            cv.waitKey(1000)
        self.setEnabled(True)
        cv.destroyAllWindows()
        self.q4_1 = True

        pass

    def matchKeypoints(self):
        if not self.q4_1:
            self.keypoints_Q4()
        desc_1, desc_2 = self.descriptors[:2]
        key_1, key_2 = self.keypoints[:2]
        img_1, img_2 = self.images_Q4[:2]

        raw_matches = self.matcher.knnMatch(desc_2, desc_1, k=2)

        point_2, point_1, keypoint_pairs = utils.filter_matches(key_2, key_1, raw_matches)

        if len(point_1) >=4:
            H, status = cv.findHomography(point_2, point_1, cv.RANSAC, 5.0)
            print('{} / {}  inliers/matched'.format(np.sum(status), len(status)))
        else:
            H, status = None, None
        self.homography = H

        image_match = utils.explore_match(img_2, img_1, keypoint_pairs, status, H)
        self.q4_2 = True
        cv.namedWindow("Match Key Point", cv.WINDOW_GUI_EXPANDED)
        cv.imshow("Match Key Point", image_match)
        cv.waitKey(1000)
        pass

    def warpImage(self):
        if not self.q4_2:
            self.matchKeypoints()
            cv.destroyAllWindows()
        img1 = self.images_Q4[0]
        img2 = self.images_Q4[1]

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if self.homography is not None:
            warp_image = cv.warpPerspective(img2, self.homography, (w1+w2, max(h1, h2)))
            warp_image[:h1, :w1] = img1
            cv.namedWindow("Warp Image", cv.WINDOW_GUI_EXPANDED)
            cv.imshow("Warp Image", warp_image)
            cv.waitKey(1000)
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(500, 150, 500, 300)
    window.show()
    sys,exit(app.exec_())

