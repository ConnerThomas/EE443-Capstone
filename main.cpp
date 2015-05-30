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

/// Global Variables
Mat src; Mat hsv;
Mat mask;
Mat hist = Mat::zeros(200, 320, CV_8UC3);
Scalar maskm;

int hBins = 12; int sBins = 12;
Mat backproj;
const char* window_image = "Source image";

/// Function Headers
void Hist_and_Backproj( );


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
//    int hsize = 16;
    //histogram value range
//    float hranges[] = {0,180};
//    const float* phranges = hranges;
    //initializes Mats
//    Mat hue = Mat::zeros(200, 320, CV_8UC3);
    //array of index pairs for mixchannels
//    int chPRE[] = {0, 0};
    
    // Trackwindow starts as entire image
    Rect trackWindow = Rect(0, 0, 640, 480);
        
    // calculate the hist of the training image
    //loads and displays training image
    src = imread( "orange.jpg", 1 );    
    cvtColor( src, hsv, COLOR_BGR2HSV );
        
    namedWindow( "Mask", WINDOW_NORMAL );
    namedWindow( "BackProj", WINDOW_NORMAL );
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
    
    Hist_and_Backproj();
    
    Mat hueF, backprojF = Mat::zeros(200, 320, CV_8UC3);
    float hranges[] = {0,180};
    const float* phranges = hranges;
    
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
        hueF.create(imgHSV.size(), imgHSV.depth());
        mixChannels(&imgHSV, 1, &hueF, 1, ch, 1);

        
        // need a hist of the obj by the time we are here
        calcBackProject(&hueF, 1, 0, hist, backprojF, &phranges);
//        backprojF &= mask;
        
        // CAMSHIFT TRACKING WE WANT THIS
        RotatedRect trackBox = CamShift(backprojF, trackWindow,
                            TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
        
        // this should run with a set track window size
        if( trackWindow.area() <= 1 )
        {
            int cols = backprojF.cols, rows = backprojF.rows, r = (MIN(cols, rows) + 5)/6;
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

/**
 * @function Hist_and_Backproj
 */
void Hist_and_Backproj( )
{
    
  // Fill and get the mask
  // seed is center 
  Point seed = Point( src.rows / 2 , src.cols / 2 );
  
  // mean stuff
//  vector<Mat> channels;
//  split(src, channels);
//  Scalar m = mean(channels[0]);
//  printf("mean of unmasked: %f\n", m[0]);
 
  int newMaskVal = 255;
  Scalar newVal = Scalar( 120, 120, 120 );

  int connectivity = 8;
  int flags = connectivity + (newMaskVal << 8 ) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;

  Mat mask2 = Mat::zeros( src.rows + 2, src.cols + 2, CV_8UC1 );
  floodFill( src, mask2, seed, newVal, 0, Scalar( 130,130,130 ), Scalar( 130,130,130 ), flags );
  
  //imshow("test", mask2);
  
  mask = mask2( Range( 1, mask2.rows - 1 ), Range( 1, mask2.cols - 1 ) );

  // open and close the mask to fill in gaps
  erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(8,8)) );
  dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(8,8)) );

  dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(8,8)) );
  erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(8,8)) );
  
//  maskm = mean(channels[0], mask); //LOL
//  printf("mean of masked: %f\n", maskm[0]);
  
  imshow( "Mask", mask );
  //imshow( "Masked Img", hsv);
  moveWindow("Mask", 200,200);
  resizeWindow("Mask",200,200);
 
  int h_bins = 10; int s_bins = 5;
  //now uses trackbar values for histogram bins
  int histSize[] = { h_bins, s_bins };

  float h_range[] = { 0, 179 };
  float s_range[] = { 0, 255 };
  const float* ranges[] = { h_range, s_range };

  int channels[] = { 0, 1 };

  /// Get the Histogram and normalize it
  calcHist( &hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false );

  normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

  /// Get Backprojection

  calcBackProject( &hsv, 1, channels, hist, backproj, ranges, 1, true );
  
  //TODO calculate back projection on full image, see what the changes in histogram values do
  //could get live camera, or just still image test

  /// Draw the backproj
  imshow( "BackProj", backproj & mask );
  resizeWindow("BackProj",200,200);

}