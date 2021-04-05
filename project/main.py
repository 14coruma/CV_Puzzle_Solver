#!/usr/bin/python3
import cv2 as cv
import classifier

if __name__ == "__main__":
    model = classifier.train('./data')

    # Video Capture code taken from https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
    cv.namedWindow("preview")
    vc = cv.VideoCapture(0)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        # Resize frame
        center = frame.shape[0] / 2, frame.shape[1] / 2
        size = min(frame.shape[0], frame.shape[1])
        x,y = center[1] - size/2, center[0] - size/2
        frame = frame[int(y):int(y+size), int(x):int(x+size)]
        frame = cv.resize(frame, (256,256))
        # Predict puzzle type
        label = classifier.predict(model, frame)[0]

        # Display matched features
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img = cv.imread('./data/{0}/{0}_0.png'.format(label), 0)
        orb = cv.ORB_create()
        kp1, des1 = orb.detectAndCompute(frame, None)
        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x: x.distance)
        img2 = cv.drawMatches(frame, kp1, img, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv.imshow("preview", img2)
        rval, frame = vc.read()
        key = cv.waitKey(200)
        if key == 27: break # exit on ESC
    cv.destroyWindow("preview")
