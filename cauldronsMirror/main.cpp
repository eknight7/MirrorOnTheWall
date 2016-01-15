#include <iostream>
#include <string.h>
#include "GroggyDetector.h"
#include "GlobalVariables.h"

using namespace cv;

int main() {

    //Capture input video
    VideoCapture vcap(CAMERA_ID);
    Mat curFrame;
    vcap.open(CAMERA_ID);
    if (!vcap.isOpened())
        cerr << "Error opening camera device #" << CAMERA_ID << endl;
    // Detect if you are sleepy!
    GroggyDetector groggyDetector;

    #if DISPLAY
        string inputWindowName = "Video Input";
        namedWindow(inputWindowName, CV_WINDOW_AUTOSIZE);
    #endif

    while (true) {
        // Camera MUST be open
        assert(vcap.isOpened());
        vcap >> curFrame;

        Mat flippedFrame;
        flip(curFrame, flippedFrame, 1);

        groggyDetector.DetectGrogginess(flippedFrame);
        #if DISPLAY
            imshow(inputWindowName, flippedFrame);
        #endif

        if (waitKey(33) == 27) // ESC key
            break;
    }

    #if DISPLAY
        destroyAllWindows();
    #endif
    
    return 0;
}