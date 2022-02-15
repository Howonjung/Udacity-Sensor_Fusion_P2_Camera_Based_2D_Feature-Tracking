/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        DataFrame frame;
        frame.cameraImg = imgGray;
        
        // // If dataBuffer.size() hasn't reached to dataBufferSize, push image into data frame buffer
        // if (dataBuffer.size() < dataBufferSize)
        //     dataBuffer.push_back(frame);
        // // otherwise, erase first element in dataBuffer and add next frame at the end
        // else{
        //     std::vector<DataFrame>::iterator it = dataBuffer.begin();
        //     dataBuffer.erase(it);
        //     dataBuffer.push_back(frame);
        // }

        // If dataBuffer.size() has same size as dataBufferSize, 
        // erase first element in dataBuffer and add next frame at the end
        if (dataBuffer.size() >= dataBufferSize){
            std::vector<DataFrame>::iterator it = dataBuffer.begin();
            dataBuffer.erase(it);
        }

        dataBuffer.push_back(frame);

        // cout << "dataBuffer.size() " << dataBuffer.size() << "\n";
        
        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        // Available detectorType options: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        string detectorType = "FAST";
        // double detKeypoingTime, descriptorExtractTime;
        // int matchSize;

        if (detectorType.compare("SHITOMASI") == 0)
            detKeypointsShiTomasi(keypoints, imgGray, false);
        else if(detectorType.compare("HARRIS") == 0)
            detKeypointsHarris(keypoints, imgGray, false);
        else if(detectorType.compare("FAST") == 0)
            detKeypointsFAST(keypoints, imgGray, false);
        else if(detectorType.compare("BRISK") == 0)
            detKeypointsBRISK(keypoints, imgGray, false);
        else if(detectorType.compare("ORB") == 0)
            detKeypointsORB(keypoints, imgGray, false);
        else if(detectorType.compare("AKAZE") == 0)
            detKeypointsAKAZE(keypoints, imgGray, false);
        else if(detectorType.compare("SIFT") == 0)
            detKeypointsSIFT(keypoints, imgGray, false);
        else{
            cout <<"detectorType: " << detectorType << " is not in available options " << "\n";
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        vector<cv::KeyPoint> filteredKeypoints; // create empty feature list for current image
        bool bFocusOnVehicle = true;
        bool bFocusOnVehicleVis = false;
        cv::Rect vehicleRect(535, 180, 180, 150);
        // Select keyPoint only if it's located in pre-defined rectangle area and add in filteredKeypoints vector
        if (bFocusOnVehicle)
        {   
            // Method 1
            // int xMin,xMax,yMin,yMax;
            // xMin = 535;
            // xMax = 535 + 180;
            // yMin = 180;
            // yMax = 180 + 150;
            // for (auto it=keypoints.begin(); it<keypoints.end(); it++){
            //     if (( (*it).pt.x > xMin && (*it).pt.x < xMax ) && ((*it).pt.y > yMin && (*it).pt.y < yMax )){
            //         filteredKeypoints.push_back(*it);
            //     }
            // }
            // // copy filteredKeypoints vector into keypoints vector
            // keypoints.clear();
            // keypoints.assign( filteredKeypoints.begin(), filteredKeypoints.end() ); 
            // filteredKeypoints.clear();

            // Method 2
            // 람다 함수 Ref: https://hwan-shell.tistory.com/84
            // [] 캡쳐 블록 (사용시 외부 변수를 캡쳐해 람다 몸통에서 사용 가능)
            // () 전달 인자
            // -> 반환 타입
            // {} 함수 몸통

            // Rect::contains() 인자로 전달된 pt점이 사각형 내부에 있으면 true를 반환
            keypoints.erase(std::remove_if(keypoints.begin(),
                                keypoints.end(),
                                [&vehicleRect](const cv::KeyPoint& kpt)-> bool 
                                       { return !vehicleRect.contains(kpt.pt); }), 
                 keypoints.end());


        }
        cout << "filteredKeypoints size: " << keypoints.size() << "\n";

        // Visualize filteredKeypoints on image
        if(bFocusOnVehicleVis){
            cv::Mat visImage1 = imgGray.clone();
            cv::drawKeypoints(imgGray, keypoints, visImage1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName1 = detectorType + " after keypoint filtering";
            cv::namedWindow(windowName1, 2);
            imshow(windowName1, visImage1);
            cv::waitKey(0);
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        string descriptorType = "BRISK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorDataType = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
            
            // change descriptorDataType into DES_HOG when descriptorType is SIFT.
            if(descriptorType.compare("SIFT") == 0){ descriptorDataType = "DES_HOG"; }

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorDataType, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = detectorType + "-"+ descriptorType + " Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }
        // cout<<fixed;
        // cout.precision(2);
        // cout <<"====================================================" <<endl;
        // cout << "detKeypoingTime, descriptorExtractTime "<< detKeypoingTime*1000.0 <<" / " << descriptorExtractTime*1000.0 << endl;
        // cout << "matchSize " << matchSize <<endl;
        // cout <<"====================================================" <<endl;
    } // eof loop over all images

    return 0;
}