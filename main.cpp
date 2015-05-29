/* 
 * File:   main.cpp
 * Author: ee443
 *
 * Created on May 19, 2015, 10:02 PM
 */
#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <ctype.h>
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main (int argc, char** argv) {
    VideoCapture cap(0);
    
    if( !cap.isOpened() ) {
        cout << "Cannot open the webcam" << endl;
        return -1;
    }
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    
    // Find the histogram of the loaded image
    
    //histogram bins
    int hsize = 16;
    //histogram value range
    float hranges[] = {0,180};
    const float* phranges = hranges;
    //initializes Mats
    Mat orangeDef, huePRE, hue, maskPRE, mask, orHSV, hist, histimg, backproj = Mat::zeros(200, 320, CV_8UC3);
    //array of index pairs for mixchannels
    int chPRE[] = {0, 0};
    
    // Trackwindow starts as entire image
    Rect trackWindow = Rect(0, 0, 640, 480);
        
    // calculate the hist of the training image
    //loads and displays training image
    orangeDef = imread("apple.jpg", CV_LOAD_IMAGE_COLOR);
    
    // replace all of this
    /*
    //converts image to HSV and stores
    cvtColor(orangeDef, orHSV, COLOR_BGR2HSV);
    //reshapes existing Mat to specified sizes
    huePRE.create(orHSV.size(), orHSV.depth());
    mixChannels(&orHSV, 1, &huePRE, 1, chPRE, 1);
    //creates mask based on HSV thresholds
    inRange(orHSV, Scalar(0, 0, 0),
                    Scalar(180, 256, 256), maskPRE);
    
    
    //computes and normalizes histogram from 0 to 255 (use equalize instead...?)
    //mask input is optional, and decides which values to be included in histogram
    calcHist(&huePRE, 1, 0, maskPRE, hist, 1, &hsize, &phranges);
    //normalizes histogram from input mask when NORM_MINMAX type
    normalize(hist, hist, 0, 255, NORM_MINMAX,-1,Mat());

    //compute back projection
    calcBackProject( &huePRE, 1, 0, hist, backproj, &phranges, 1, true);
    
    int lower, higher, mean, total;
    */
    
    // main video tracking while loop
    while (true) {
        Mat imgOriginal;
        
        // get the frame
        bool bSuccess = cap.read(imgOriginal);
        
        if (!bSuccess) {
            cout << "Cannot read frame from video stream" << endl;
            break;
        }
        
        // FRAME PROCESSING BLOCK: 
        // convert video frame to HSV
        Mat imgHSV;        
        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);        
        // mask the frame to useful ranges
//        inRange(imgHSV, Scalar(0, 100, 0),
//                    Scalar(70, 200, 256), mask);
        int ch[] = {0, 0};
        hue.create(imgHSV.size(), imgHSV.depth());
        mixChannels(&imgHSV, 1, &hue, 1, ch, 1);

        
        // need a backproj Mat by the time we are here
        calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
//        backproj &= mask;
        
        // CAMSHIFT TRACKING WE WANT THIS
        RotatedRect trackBox = CamShift(backproj, trackWindow,
                            TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
        if( trackWindow.area() <= 1 )
        {
            int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
            trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                               trackWindow.x + r, trackWindow.y + r) &
                          Rect(0, 0, cols, rows);
        }
        ellipse( imgOriginal, trackBox, Scalar(0,0,255), 3, LINE_AA );        
             
        imshow( "CamShift Demo", imgOriginal );
        moveWindow("Camshift Demo", 300, 300);
        
        // handle user input
        if (waitKey(15) == 27) {
            cout << "esc key pressed by user" << endl;
            break;
        }

    }
    
    
    return 0;
}
