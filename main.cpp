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
    
    
    // Trackwindow starts as entire image
    Rect trackWindow = Rect(0, 0, 640, 480);
        
    // calculate the hist of the training image
    //loads and displays training image
    src = imread( "orange.jpg", 1 );    
    cvtColor( src, hsv, COLOR_BGR2HSV );
        
    namedWindow( "Mask", WINDOW_NORMAL );
    namedWindow( "BackProjF", WINDOW_NORMAL );
    namedWindow( "BackProj", WINDOW_NORMAL );
    // replace all of this
    
    
    Hist_and_Backproj();
    
    Mat hueF = Mat::zeros(200, 320, CV_8UC3);
    Mat backprojF;
    float h_rangeF[] = { 0, 179 };
    float s_rangeF[] = { 0, 255 };
    const float* Franges[] = { h_rangeF, s_rangeF };
    
    printf("channels of hist after gen: %d\n", hist.channels());
    
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
        int ch[] = {0,0, 1,1};
        hueF.create(imgHSV.size(), CV_8UC3);
        mixChannels(&imgHSV, 2, &hueF, 2, ch, 2);
        
        int chF[] = {0, 1};
        //printf("channels of hueF: %d\n", hueF.channels());
        //printf("dims of hist: %d\n", hist.dims);
        
        // need a hist of the obj by the time we are here
        calcBackProject( &hueF, 1, chF, hist, backprojF, Franges, 1, true );
        // calcBackProject( &hsv, 1, channels, hist, backproj, ranges, 1, true );
//        backprojF &= mask;
        
        //printf("testback");
        
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
        imshow( "BackProjF", backprojF );
        imshow( "BackProj", backproj );
        
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

  printf("channels of hsv: %d\n", hsv.channels());
  //printf("dims of hist: %d\n", hist.dims);
  
  /// Get the Histogram and normalize it
  calcHist( &hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false );
  printf("channels of hist during gen: %d\n", hist.channels());

  normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

  /// Get Backprojection
  
  printf("channels of hsv: %d\n", hsv.channels());

  calcBackProject( &hsv, 1, channels, hist, backproj, ranges, 1, true );
  
  //TODO calculate back projection on full image, see what the changes in histogram values do
  //could get live camera, or just still image test

  /// Draw the backproj
  imshow( "BackProj", backproj & mask );
  resizeWindow("BackProj",200,200);

}