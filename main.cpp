/* 
 * File:   main.cpp
 * Authors: Conner Thomas & Nick Morello
 * 
 * EE 443 Capstone Design Project
 *
 * NOTE: CAMERA SETTINGS: B117 C148 S202 G164 S145 BC0 Aperture priority
 * Brightness, Contrast, Saturation, Gain, Sharpness, Backlight Compensation
 * Using Logitech C920 HD Pro
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

static void help()
{
    cout << "\n\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tb - switch to/from backprojection view\n"
            "\tBACKSPACE - clear the tracking history\n";
}

/// Global Variables
Mat src; Mat hsv;
Mat mask;
Mat hist = Mat::zeros(200, 320, CV_8UC3);
Scalar maskm;

// camera frame resolution
int camW = 640;
int camH = 480;

// bounding rect area limits for threshold of tracking = true
int lowerSize = 500;
int upperSize = 62500;

bool backprojMode = false;

// H/S histogram settings
int h_bins = 15; int s_bins = 10;
Mat backproj;
const char* window_image = "Source image";

int iLastX = -1;
int iLastY = -1;
Mat imgTmp;
Mat imgLines;

bool searchMode = true;
int searchQuad = 0;

RotatedRect trackBox;

/// Function Headers
void Hist_and_Backproj( );
//void clearHist( int state, *void);


int main (int argc, char** argv) {
    
    VideoCapture cap(0);
    
    if( !cap.isOpened() ) {
        cout << "Cannot open the webcam" << endl;
        return -1;
    }
    cap.set(CV_CAP_PROP_FRAME_WIDTH, camW);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, camH);
    
    help();
    
    //initialize history Mat
    cap.read(imgTmp);
    imgLines = Mat::zeros(imgTmp.size(),CV_8UC3);;
   
    // Trackwindow starts as entire image
    Rect trackWindow = Rect(0, 0, camW, camH);
        
    // calculate the hist of the training image
    //loads and displays training image
    //LOAD IMAGE HERE
    src = imread( "lime2Adj.jpg", 1 );  
    //LOAD IMAGE HERE
    
    //TODO: Load command line input file location
    
    
    cvtColor( src, hsv, COLOR_BGR2HSV );
    //namedWindow("CamShift Demo", WINDOW_NORMAL);
    //createButton("Clear tracking history", clearHist*);
    namedWindow("Reference Image", WINDOW_NORMAL);
    namedWindow( "Mask", WINDOW_NORMAL );
    namedWindow( "BackProj", WINDOW_NORMAL );
    
    Hist_and_Backproj();
    
    Mat hueF = Mat::zeros(200, 320, CV_8UC3);
    Mat backprojF;
    Mat bpMask;
    float h_rangeF[] = { 0, 179 };
    float s_rangeF[] = { 0, 255 };
    const float* Franges[] = { h_rangeF, s_rangeF };
    
    //printf("channels of hist after gen: %d\n", hist.channels());
    
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
        
        //gaussian blur input image to reduce noise
        GaussianBlur(imgHSV, imgHSV, Size(5,5), 0.5);
        
        // mask the frame to useful ranges
//        inRange(imgHSV, Scalar(0, 100, 0),
//                    Scalar(70, 200, 256), mask);
        int ch[] = {0,0, 1,1};
        hueF.create(imgHSV.size(), CV_8UC3);
        mixChannels(&imgHSV, 2, &hueF, 2, ch, 2);
        
        int chF[] = {0, 1};
        //printf("channels of hueF: %d\n", hueF.channels());
        //printf("dims of hist: %d\n", hist.dims);
        
        // need a hist of the object by the time we are here
        calcBackProject( &hueF, 1, chF, hist, backprojF, Franges, 1, true );
        
        //blurs to reduce possible noise [instead done on original HSV]
//        GaussianBlur(backprojF, backprojF, Size(5,5), 0.5);
        
        //open operation to get rid of noise "specs"
        erode(backprojF, backprojF, getStructuringElement(MORPH_ELLIPSE, Size(3,3)) );
        dilate(backprojF, backprojF, getStructuringElement(MORPH_ELLIPSE, Size(3,3)) );
        
        double minF;
        double maxF;
        minMaxLoc(backprojF, &minF, &maxF);
        //printf("max backprojF %f\n", maxF);
        
        //relative threshold.. doesn't work well if no object in frame
        //could be fixed: assume object first in frame and STORE max
        //inRange(backprojF, (int)(0.25*maxF), 255, bpMask);
        
        inRange(backprojF, 80, 255, bpMask);
        backprojF &= bpMask;
        
        if (searchMode) {
//        
//        iterate track window in quadrants across image
//        once detection threshold met, "find image" and enter tracking mode
//        detection: aspect ratio between a and b, size not greater than 250x250
            
            switch(searchQuad) { //left to right, top to bottom
                case 0: //Q1
                    trackWindow = Rect(0,0,camW/2,camH/2);
                    searchQuad = 1;
                    break;
                case 1: //Q2
                    trackWindow = Rect(camW/2,0,camW,camH/2);
                    searchQuad = 2;
                    break;
                case 2: //Q3
                    trackWindow = Rect(0,camH/2,camW/2,camH);
                    searchQuad = 3;
                    break;
                case 3: //Q4
                    trackWindow = Rect(camW/2,camH/2,camW,camH);
                    searchQuad = 0;
                    break;
                default:
                    break;
            }
            
            trackBox = CamShift(backprojF, trackWindow,
                            TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
            
            //printf("Area of box: %d\n", trackBox.boundingRect().area());
            
            if (trackBox.boundingRect().area() > lowerSize &&
                trackBox.boundingRect().area() < upperSize) {
                cout << "Object detected, beginning tracking" << endl;
                searchQuad = 0;
                searchMode = false;
                circle(imgLines, trackBox.center, 5, Scalar(0,255,0), -1);
            }
            
        } else {
            // normal tracking
            trackBox = CamShift(backprojF, trackWindow,
                                TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));

            // this should run with a set track window size
            if( trackWindow.area() <= 1 )
            {
                int cols = backprojF.cols, rows = backprojF.rows, r = (MIN(cols, rows) + 5)/6;
                trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                   trackWindow.x + r, trackWindow.y + r) &
                              Rect(0, 0, cols, rows);
            }

            //threshold to enter search mode
            //use aspect ratio of trackbox
            //possibly check for image shape, implement later
            //print "object lost", enter search mode
            if (trackBox.boundingRect().area() < lowerSize ||
                    trackBox.boundingRect().area() > upperSize) {
                cout << "Object not found, entering search mode" << endl;
                searchMode = true;

                if (iLastX >= 0 && iLastY >= 0) {
                    circle(imgLines, Point(iLastX, iLastY), 5, Scalar(0,0,255), -1);
                }
                iLastX = -1;
                iLastY = -1;
            }
        }
        
        //replaces the "standard" live video with view of live back projection
        if( backprojMode ) {
                cvtColor( backprojF, imgOriginal, COLOR_GRAY2BGR );
        }
        
        if (!searchMode) { //ensure object wasnt lost before drawing track lines
            //ellipse( imgOriginal, trackBox, Scalar(0,0,255), 3, LINE_AA );  
            rectangle(imgOriginal, trackBox.boundingRect(), Scalar(0,255,0));
            int posX = trackBox.center.x;
            int posY = trackBox.center.y;

            //if the object is far enough away from the previous point
            if (abs(posX - iLastX) > 5 && abs(posY - iLastY)) {

                //and the object has already been detected at least once
                if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0) {
                    //draw line from previous position to new position
                    line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(255,0,255), 2 );     
                }

                //update position buffer
                iLastX = posX;
                iLastY = posY;

            }
        }
        
        //add object history to output image, then flip to accurately mirror
        imgOriginal = imgOriginal + imgLines;
        flip(imgOriginal,imgOriginal,1);
             
        imshow( "Object Tracking", imgOriginal );
        imshow( "BackProj", backproj );
        
        
        char c = (char)waitKey(10);
        if( c == 27 ) { //ends program
            cout << "ESC key pressed by user, closing" << endl;
            break;
        }
        switch(c)
        {
        case 'b': //switches to/from back projection display mode
            backprojMode = !backprojMode;
            break;
        case 8: //clears the tracking history
            cout << "backspace key pressed by user, clearing history" << endl;
            imgLines = Mat::zeros(imgLines.size(),CV_8UC3);; 
            if (!searchMode)
                circle(imgLines, trackBox.center, 5, Scalar(0,255,0), -1);
            break;
        default:
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
  //Point seed = Point( src.rows / 2 , src.cols / 2 );
  //printf("x: %d y: %d\n", seed.x, seed.y);
  // seed is top left
    Point seed = Point(0,0);
  
  // mean stuff
//  vector<Mat> channels;
//  split(src, channels);
//  Scalar m = mean(channels[0]);
//  printf("mean of unmasked: %f\n", m[0]);
 
  int newMaskVal = 255;
  Scalar newVal = Scalar( 120, 120, 120 );

  int connectivity = 8;
  int flags = connectivity + (newMaskVal << 8 ) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;

  //mask size requirement for floodFill
  Mat mask2 = Mat::zeros( src.rows + 2, src.cols + 2, CV_8UC1 );
  int lo = 60;
  int hi = 60;
  //floodfill starts at the top left, and "selects" the monochromatic background
  //it then creates a binary mask of this background, which we inverse to mask the object
  floodFill( src, mask2, seed, newVal, 0, Scalar( 45,lo,lo ), Scalar( 45,hi,hi ), flags );
  
  imshow("Reference Image", src);
  moveWindow("Reference Image", 400,600);
  resizeWindow("Reference Image", 200,200);
  
  //reverts mask back to size of input frame
  mask = mask2( Range( 1, mask2.rows - 1 ), Range( 1, mask2.cols - 1 ) );
  //inverts mask
  mask = 255 - mask;

  // open and close the mask to fill in gaps
  erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(8,8)) );
  dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(8,8)) );

  dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(8,8)) );
  erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(8,8)) );
  
  //mean no longer needed, kept for reference
//  maskm = mean(channels[0], mask); //easy masked mean
//  printf("mean of masked: %f\n", maskm[0]);
  
  imshow( "Mask", mask );
  moveWindow("Mask", 400,400);
  resizeWindow("Mask",200,200);
  
  //uses global histogram bin numbers
  //TODO: possibly read values as user input, or use trackbar to refine tracking
  int histSize[] = { h_bins, s_bins };

  //histogram bin range parameters
  float h_range[] = { 0, 179 };
  float s_range[] = { 0, 255 };
  const float* ranges[] = { h_range, s_range };

  //channels for histogram, makes it use H and S from HSV
  int channels[] = { 0, 1 };

//  printf("channels of hsv: %d\n", hsv.channels());
  //printf("dims of hist: %d\n", hist.dims);
  
  /// Get the Histogram and normalize it
  calcHist( &hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false );
  //calcHist( &hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false ); // doesn't use mask
//  printf("channels of hist during gen: %d\n", hist.channels());

  normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

  /// Get Backprojection
  
//  printf("channels of hsv: %d\n", hsv.channels());

  calcBackProject( &hsv, 1, channels, hist, backproj, ranges, 1, true );
  
  //TODO calculate back projection on full image, see what the changes in histogram values do
  //could get live camera, or just still image test

  /// Draw the backproj
  imshow( "BackProj", backproj & mask );
  resizeWindow("BackProj",200,200);

}