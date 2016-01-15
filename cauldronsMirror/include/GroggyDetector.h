#ifndef GROGGYDETECTOR_H
#define GROGGYDETECTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

class GroggyDetector
{
    public:
        GroggyDetector();
        virtual ~GroggyDetector();
        void DetectFaces(Mat &frame);
        void DetectGrogginess(Mat &frame);
        bool CheckEyeState(Mat &frame, Rect &eyeRegion);
        void SetInputWindowName(string name);
    protected:
        void Dilation(Mat &image, Mat &result);
    private:
        Mat curFrame;
        Mat grayFrame;
        string inputWindowName;
        vector< Rect_<int> > faceRects;
        // Face cascade classifier
        CascadeClassifier faceCascade;
        // Parameters for cascade classifier of face
        double faceScaleFactor;
        double faceMinNbs;
        Size faceMinSize;
        // Eye cascade classifier
        CascadeClassifier eyeCascade;
        // Parameters for cascade classifier of eyes
        double eyeScaleFactor;
        double eyeMinNbs;
        Size eyeMinSize;
};

#endif // GROGGYDETECTOR_H