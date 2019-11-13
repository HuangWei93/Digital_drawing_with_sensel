import numpy as np
import cv2
import cv2.aruco as aruco
from calib import *
 
cap = cv2.VideoCapture(0)
ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')
 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = undistort(gray, mtx, dist)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if(len(corners) == 4):
        print("Detected four aruco markers")
        idx = sorted(range(len(ids)), key=lambda k: ids[k])
        corners = [corners[i] for i in idx]
        ids = [ids[i] for i in idx]
        points_idx = [0,1,3,2]
        h, w = 550, 950
        vertices = []
        for i, corner in enumerate(corners):
            print("id: %d x = %d y = %d" % (i, corner[0, points_idx[i], 0],corner[0, points_idx[i], 1] ))
            vertices.append(corner[0, points_idx[i], :])
        src = [vertices[i] for i in [3,2,0,1]]
        src = np.float32(src)
        dst = np.float32([[w, h],       # br
                  [0, h],       # bl
                  [0, 0],       # tl
                  [w, 0]])      # tr
        M = cv2.getPerspectiveTransform(src, dst)
        gray = cv2.warpPerspective(gray, M, (w, h), flags=cv2.INTER_LINEAR)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
