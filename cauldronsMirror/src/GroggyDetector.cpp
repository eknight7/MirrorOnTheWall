#include "GlobalVariables.h"
#include "GroggyDetector.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace cv;
using namespace std;

GroggyDetector::GroggyDetector()
{
    // Constructor
    faceScaleFactor = 1.1;
    faceMinNbs = 5;
    faceMinSize = Size(30, 30);
    eyeScaleFactor = 1.1;
    eyeMinNbs = 10;
    eyeMinSize = Size(20, 20);
}

GroggyDetector::~GroggyDetector()
{
    // Destructor
}

void GroggyDetector::SetInputWindowName(string name)
{
    inputWindowName = name;
}

void GroggyDetector::DetectFaces(Mat &frame)
{
    curFrame = frame.clone();
    // Convert to grayscale
    cvtColor(curFrame, grayFrame, CV_BGR2GRAY);
    if (!grayFrame.empty()) {
        // Load face cascade classifier
        faceCascade.load(FACE_CASCADE);
        // Find the faces in the frame
        faceCascade.detectMultiScale(grayFrame, faceRects, faceScaleFactor,
            faceMinNbs, CV_HAAR_SCALE_IMAGE, faceMinSize);
    }
}

void GroggyDetector::Dilation(Mat &image, Mat &result)
{
    Mat element = getStructuringElement(DILATION_TYPE,
                                        Size(2 * DILATION_SIZE + 1, 2 * DILATION_SIZE + 1),
                                        Point(DILATION_SIZE, DILATION_SIZE));
    dilate(image, result, element);
}

bool GroggyDetector::CheckEyeState(Mat &frame, Rect &eyeRegion)
{
    // If we can find an eye in this region its open!
    Mat curEyeRegion = frame(eyeRegion);
    vector< Rect_<int> > eyeRects;
    eyeCascade.detectMultiScale(curEyeRegion, eyeRects, eyeScaleFactor,
        eyeMinNbs, CV_HAAR_SCALE_IMAGE, eyeMinSize);
    return eyeRects.size() > 0;
}

void GroggyDetector::DetectGrogginess(Mat &frame)
{
    DetectFaces(frame);
    
    // Need atleast 1 face
    if (faceRects.size() < 1)
        cerr << "No faces!" << endl;
    else {
        // Load eye cascade classifier
        eyeCascade.load(EYE_CASCADE);

        cout << "Groggy face count = " << faceRects.size() << endl;
        for (int i = 0; i < faceRects.size(); i++) {
            Rect curFaceRect = faceRects[i];

            // Crop out eye regions from face
            int eyeLeft = curFaceRect.x + curFaceRect.width / 8.0;
            int eyeTop = curFaceRect.y + curFaceRect.height / 4.0;
            int eyeRight = curFaceRect.x + 7.0 * curFaceRect.width / 8.0;
            int eyeBottom = curFaceRect.y + curFaceRect.height / 2.0;

            #if DEBUG
                cout << "eyeLeft = " << eyeLeft << endl;
                cout << "eyeRight = " << eyeRight << endl;
                cout << "eyeTop = " << eyeTop << endl;
                cout << "eyeBottom = " << eyeBottom << endl;

                cout << "faceLeft = " << curFaceRect.x << endl;
                cout << "faceRight = " << curFaceRect.y << endl;
                cout << "faceWidth = " << curFaceRect.width << endl;
                cout << "faceHeight = " << curFaceRect.height << endl;
            #endif

            Rect eyeRegionRect = 
                Rect(eyeLeft, eyeTop, eyeRight - eyeLeft, eyeBottom - eyeTop);

            // Select both eyes separtely
            Rect eyeRegionLeft = Rect(eyeLeft, eyeTop, eyeRegionRect.width/2,
                                        eyeRegionRect.height);
            Rect eyeRegionRight = Rect(eyeLeft + eyeRegionRect.width/2,
                                        eyeTop, eyeRegionRect.width/2,
                                        eyeRegionRect.height);

            // Convert eye region to grayscale
            Mat eyeRegionGray;
            cvtColor(frame(eyeRegionRect), eyeRegionGray, CV_BGR2GRAY);

            // For each eye, check state
            bool eyeStateLeft = CheckEyeState(frame, eyeRegionLeft);
            bool eyeStateRight = CheckEyeState(frame, eyeRegionRight);

            if (eyeStateLeft && eyeStateRight)
                cout << "AWAKE!" << endl;
            else if (eyeStateLeft || eyeStateRight)
                cout << "You winking at me?" << endl;
            else
                cout << "Rise and shine sleepy head, half the town is probably dead!" << endl;


            /*
            // Binarize at threshold
            Mat eyeRegionBin;
            threshold(eyeRegionGray, eyeRegionBin, EYE_REGION_THRESH, MAX_VAL, 0);

            // Dilation
            Mat eyeRegionDil;
            Dilation(eyeRegionBin, eyeRegionDil);

            // Invert the binary image
            Mat eyeRegionInv;
            eyeRegionInv = Scalar::all(255) - eyeRegionDil;

            // Invert once more
            Mat eyeRegionForSpecs;
            eyeRegionForSpecs = Scalar::all(255) - eyeRegionInv;

            // Apply boundry fill on each eye region


            imshow("eyeRegionGray", eyeRegionGray);
            imshow("eyeRegionBin", eyeRegionBin);
            imshow("eyeRegionDil", eyeRegionDil);
            imshow("eyeRegionInv", eyeRegionInv);
            imshow("eyeRegionForSpecs", eyeRegionForSpecs);

            rectangle(frame, eyeRegionRect, CV_RGB(0, 255, 0), 1);
            imshow("frame", frame);
            */

            rectangle(frame, eyeRegionLeft, CV_RGB(0, 255, 0), 1);
            rectangle(frame, eyeRegionRight, CV_RGB(0, 0, 255), 1);
            imshow("eyeRegions", frame);

            #if DEBUG
                while (true) {
                    if (waitKey(33) == (int)('s'))
                        cout << "printed s" << endl;
                    if (waitKey(33) == 32) // SPACE key
                        break;
                }
            #endif

            /*
            // Crop out the face region
            Mat curFace = grayFrame(curFaceRect);
            imshow("faces", curFace);
            
            // Find the eyes in the face
            vector< Rect_<int> > eyeRects;
            eyeCascade.detectMultiScale(curFace, eyeRects, eyeScaleFactor,
                eyeMinNbs, CV_HAAR_SCALE_IMAGE, eyeMinSize);
            if (eyeRects.size() > 0)
                cout << "Found # "<< eyeRects.size() << " eyes" << endl;
            for (int j = 0; j < eyeRects.size(); j++) {
                // crop out the eye region
                Mat curEye = curFace(eyeRects[j]);
                imshow("eyes", curEye);
                while (true) {
                    if (waitKey(33) == 32) // SPACE key
                        break;
                }
            }
            */
        }
    }
    
}


/*
VideoCapture cap(2);  // open the default camera
if (!cap.isOpened())  // check if we succeeded
    return -1;
namedWindow("Camera Feed", CV_WINDOW_AUTOSIZE);

// Track time elapsed for computing FPS
time_t startTime, curTime;

time(&startTime);
int numFramesCaptured = 0;
double secElapsed;
double curFPS;
double averageFPS = 0.0;

while (true) {
    // Get the current frame
    Mat frame;
    cap >> frame;
    // Show the frame
    imshow("Camera Feed", frame);

    numFramesCaptured++;
    // Get the current time and show FPS
    time(&curTime);
    double secElapsed = difftime(curTime, startTime);
    double curFPS = numFramesCaptured / secElapsed;

    cout << "FPS = " << curFPS << endl;
    # if DEBUG
    cout << "secElapsed = " << secElapsed << " secs, numFramesCaptured = " << numFramesCaptured << endl;
    #endif
    // compute running average of frames
    if (secElapsed > 0)
        averageFPS = (averageFPS * (numFramesCaptured - 1) + curFPS)
                            / numFramesCaptured;
    if (waitKey(33) == 27)
        break;
}

cout << "Average FPS = " << averageFPS << endl;

// the camera will be deninitialized automatically in
// VideoCapture destructor;
*/
