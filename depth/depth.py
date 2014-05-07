
import os
import cv2
import numpy as np
import calibrate_stereo as cal


def main():
    ii_l = cv2.imread("left.jpg")
    ii_r = cv2.imread("right.jpg")
    c = cal.StereoCalibration(input_folder="../cam_data")
    (ri_l, ri_r) = c.rectify((ii_l, ii_r))
    cv2.namedWindow("left")
    cv2.moveWindow("left", 20, 40)
    cv2.namedWindow("right")
    cv2.moveWindow("right", 700, 40)
    cv2.imshow("left", ri_l)
    cv2.imshow("right", ri_r)
    cv2.imwrite("left_rct.jpg", ri_l, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite("right_rct.jpg", ri_r, [cv2.IMWRITE_JPEG_QUALITY, 90])
    sgbm = cv2.StereoSGBM()
    sgbm.SADWindowSize = 11
    sgbm.preFilterCap = 31
    sgbm.minDisparity = 0
    sgbm.speckleWindowSize = 9
    sgbm.speckleRange = 24
    sgbm.disp12MaxDiff = 1
    sgbm.numberOfDisparities = 128
    disp = cv2.normalize(sgbm.compute(ri_l, ri_r), alpha=0, beta=255, \
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.namedWindow("Disparity")
    cv2.moveWindow("Disparity", 400, 500)
    cv2.imshow("Disparity", disp)
    cv2.imwrite("disparity.jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.waitKey(30000)
    cv2.destroyWindow("left")
    cv2.destroyWindow("right")

if __name__ == "__main__":
    main()
