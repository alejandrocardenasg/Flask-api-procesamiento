import argparse
import cv2
import math
import time
import numpy as np
from config import Models

class PoseEstimation:
    def __init__(self, mode):
        self.mode = mode
        self.models = Models()
        self.load_model()

        self.threshold = 0.1
        self.angsD = []
        self.angsQ = []

    def load_model(self):
        if self.mode == "COCO":
            self.protoFile = self.models.protoFile_coco
            self.weightsFile = self.models.weightsFile_coco
            self.nPoints = self.models.nPoints_coco
            self.pose_pairs = self.models.pose_pairs_coco
        elif self.mode == "MPI":
            self.protoFile = self.models.protoFile_mpi
            self.weightsFile = self.models.weightsFile_mpi
            self.nPoints = self.models.nPoints_mpi
            self.pose_pairs = self.models.pose_pairs_mpi
        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)

    def load_img(self, img_path):
        self.name_file = img_path.split('/')[-1]
        self.frame = cv2.imread(img_path)
        self.frame = cv2.resize(self.frame, dsize=(600, 1000))
        self.frameWidth = self.frame.shape[1]
        self.frameHeight = self.frame.shape[0]

    def distance(self, pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def getdist(self, pointsList):
        pt1, pt2 = pointsList[-2:]
        dInch = (self.distance(pt1, pt2))/100
        dCm = dInch*2.54
        d = 2*dCm/11.98
        cv2.putText(self.frame, str("{0:.2f}".format(
            d)), (pt1[0]-40, pt1[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def gradient(self, pt1, pt2):
        if pt1[1] == pt2[1]:
            y = (pt2[1]-(pt1[1]+1))/(pt2[0]-pt1[0])
        elif pt1[0] == pt2[0]:
            y = (pt2[1]-pt1[1])/(pt2[0]-(pt1[0]+1))
        else:
            y = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
        return y

    def half(self,pointsList):
        center_up = None
        center = None
        down = None
        if pointsList[1] == None:
            if pointsList[2] != None and pointsList[5] != None:
                #calcula la mitad entre 5 y 2
                center_up = ((pointsList[2][0]+pointsList[5][0])/2,(pointsList[2][1]+pointsList[5][1])/2)
        else:
            center_up = pointsList[1]
        
        if pointsList[8] != None and pointsList[11] != None:
            #calcula la mitad entre 5 y 2
            center = ((pointsList[8][0]+pointsList[11][0])/2,(pointsList[8][1]+pointsList[11][1])/2)

        if pointsList[9] == None or pointsList[12] == None:
            if pointsList[12] != None:
                down = pointsList[12]

            if pointsList[9] != None:
                down = pointsList[9]
        else:
            down = ((pointsList[9][0]+pointsList[12][0])/2,(pointsList[9][1]+pointsList[12][1])/2)
        
        if center_up and center and down:
            m1 = self.gradient(center, center_up)
            m2 = self.gradient(center, down)
            angR = np.arctan((m2-m1)/(1+(m2*m1)))  # cálculo de grados
            angD = round(math.degrees(angR))  # conversión a grados
            print("The angle Half is " + str(angD))
            self.angsD.append(angD)
            self.angsQ.append(angD)
            cv2.putText(self.frame, str(angD), 
                        (int(center[0])-40, int(center[1])-20), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)
            cv2.line(self.frame, (int(center_up[0]),int(center_up[1]+100)), (int(center[0]),int(center[1])), (0, 255, 0), 1)
            cv2.line(self.frame, (int(center[0]),int(center[1])), (int(down[0]),int(down[1])), (0, 255, 0), 1)
        else:
            error
    
    def neck(self, pointsList):
        center_n = None
        ref_line = None
        center_upn = None
        if pointsList[1] == None:
            if pointsList[2] != None and pointsList[5] != None:
                #calcula la mitad entre 5 y 2
                center_n = ((pointsList[2][0]+pointsList[5][0])/2,(pointsList[2][1]+pointsList[5][1])/2)
        else:
            center_n = pointsList[1]
        
        ref_line = (center_n[0],center_n[1]-80)#test

        if pointsList[16] != None and pointsList[15] != None:
            #calcula la mitad entre 5 y 2
            center_upn = ((pointsList[16][0]+pointsList[15][0])/2,(pointsList[16][1]+pointsList[15][1])/2)
        elif pointsList[16] == None or pointsList[15] == None:
            if pointsList[16] != None:
                center_upn = pointsList[16]

            if pointsList[15] != None:
                center_upn = pointsList[15]
        else:
            center_upn = pointsList[15]
        
        if center_upn and center_n and ref_line:
            m1 = self.gradient(center_n, center_upn)
            m2 = self.gradient(center_n, ref_line)
            angR = np.arctan((m2-m1)/(1+(m2*m1)))  # cálculo de grados
            angD = round(math.degrees(angR))  # conversión a grados
            self.angsD.append(angD)
            self.angsQ.append(angD)
            cv2.putText(self.frame, str(angD),
                        (center_n[0]-40, center_n[1]-20), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)
            cv2.line(self.frame, center_upn, center_n, (0, 255, 0), 1)
            cv2.line(self.frame, center_n, ref_line, (0, 255, 0), 1)

    def getAngle(self, points):
        pt1, pt2, pt3 = points[-3:]  # tomo los últimos 3 puntos
        m1 = self.gradient(pt2, pt1)
        m2 = self.gradient(pt2, pt3)
        angR = math.atan((m2-m1)/(1+(m2*m1)))  # cálculo de grados
        angD = abs(round(math.degrees(angR)))  # conversión a grados
        self.angsD.append(angD)
        self.angsQ.append(str(angD))
        cv2.putText(self.frame, str(angD),
                    (pt1[0]-40, pt1[1]-20), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 2)
        cv2.line(self.frame, pt1, pt2, (0, 255, 255), 1)
        cv2.line(self.frame, pt2, pt3, (0, 255, 255), 1)
            
    def getBlobresult(self):
        inpBlob = cv2.dnn.blobFromImage(self.frame, 1.0 / 255, (368, 368),
                                        (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inpBlob)
        return self.net.forward()
    
    def getAngles(self):
        print(self.angsQ)
        return self.angsQ

    def mainloop(self, verbose, filename):
        
        output = self.getBlobresult()
        H = output.shape[2]
        W = output.shape[3]

        # Empty list to store the detected keypoints
        points = []
        # ángulos
        for i in range(self.nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (self.frameWidth * point[0]) / W
            y = (self.frameHeight * point[1]) / H

            if prob > self.threshold:
                cv2.circle(self.frame, (int(x), int(y)),2, (0, 255, 255),
                        thickness=-1, lineType=cv2.FILLED)
                cv2.putText(self.frame, "{}".format(i), (int(x), int(
                    y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)
   
            try: 
                if i >= 2 and i <= 18:
                    if len(points) == 5:
                        try:
                            self.getAngle(points)
                        except:
                            self.angsQ.append("Ángulo no disponible")
                    elif len(points) == 8:
                        try:
                            self.getAngle(points)
                        except:
                            self.angsQ.append("Ángulo no disponible")
                    elif len(points) == 11:
                        try:
                            self.getAngle(points)
                        except:
                            self.angsQ.append("Ángulo no disponible")
                    elif len(points) == 14:
                        try:
                            self.getAngle(points)
                        except:
                            self.angsQ.append("Ángulo no disponible")
                        try:
                            self.half(points)
                        except:
                            self.angsQ.append("Ángulo no disponible")
                    elif len(points) == 17:
                        try:
                            self.neck(points)
                        except:
                            self.angsQ.append("Ángulo no disponible")
            except Exception as d:
                #self.angsQ.append("Ángulo no disponible")
                print("Angle not found")
        if verbose:
            cv2.imshow('imagen salida',self.frame)
            cv2.waitKey(0)
        cv2.imwrite("resultado/" + filename + ".jpg",self.frame)
        print("Angles: {}".format(self.angsD))

