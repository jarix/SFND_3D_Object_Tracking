
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"


using namespace std;

static int callCount = 0;  // for Debugging

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    cout << "Entering computeTTCCamera() .." << endl;

    std::vector<double> distRatios;

    // Get 2 pairs of keypoints and determine distance ratio 
    for (auto itr1 = kptMatches.begin(); itr1 != kptMatches.end()-1; ++itr1)
    {
        cv::KeyPoint prevKpt1 = kptsPrev.at(itr1->queryIdx);
        cv::KeyPoint currKpt1 = kptsCurr.at(itr1->trainIdx);
        
        for (auto itr2 = kptMatches.begin()+1; itr2 != kptMatches.end(); ++itr2)
        {
            double minDist = 100.0;

            cv::KeyPoint prevKpt2 = kptsPrev.at(itr2->queryIdx);
            cv::KeyPoint currKpt2 = kptsCurr.at(itr2->trainIdx);

            // Computer Distancs between 2 keypoints in each frame
            double prevDist = cv::norm(prevKpt1.pt - prevKpt2.pt);
            double currDist = cv::norm(currKpt1.pt - currKpt2.pt);

            //cout << "prevDist = " << prevDist << ", currDist = " << currDist << endl;

            if ((prevDist > std::numeric_limits<double>::epsilon()) && (currDist >= minDist) )
            {
                double distRatio = currDist / prevDist;
                distRatios.push_back(distRatio);
            }
        }
    }

    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    double dT  = 1 / frameRate;

    double medianDistRatio = calcMedianDistance(distRatios);

    TTC = -dT / (1 - medianDistRatio);

    cout << "TTC Camera = " << TTC << endl;

}



// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, 
                            std::vector<cv::KeyPoint> &kptsPrev, 
                            std::vector<cv::KeyPoint> &kptsCurr, 
                            std::vector<cv::DMatch> &kptMatches)
{   
    cout << "Entering clusterKptMatchesWithROI(), call # = " << callCount++ << endl;
    printBoundingBoxData(boundingBox, false);

    cv::Point prevKp;
    cv::Point currKp;
    std::vector<double> kptDistances; 
    double kptDist;   

    // Get keypoint locations    
    for (auto itr = kptMatches.begin(); itr != kptMatches.end(); ++itr) 
    {
        prevKp = kptsPrev[itr->queryIdx].pt;
        currKp = kptsCurr[itr->trainIdx].pt;

        // Check if current keypoint is withing bounding box
        if ((currKp.x > boundingBox.roi.x) && (currKp.x < (boundingBox.roi.x + boundingBox.roi.width)) &
            (currKp.y > boundingBox.roi.y) && (currKp.y < (boundingBox.roi.y + boundingBox.roi.height)) ) 
        {        
            kptDist = cv::norm(currKp - prevKp);
            kptDistances.push_back(kptDist);
            //cout << prevKp.x << "," << prevKp.y << "->" << currKp.x << "," << currKp.y << " = " << kptDist << endl; 
        } 
    }
    cout << "Number of valid keypoints: " << kptDistances.size() << endl;

    // Calculate mean distance:
    auto n = kptDistances.size();
    double meanDist = 0.0;
    if (n != 0) {
        meanDist  = std::accumulate(kptDistances.begin(), kptDistances.end(), 0.0) / n;
    }
    cout << "Mean distance = " << meanDist << endl;

    // Populate BoundingBox struct and ignore distances that a greater than mean
    for (auto itr = kptMatches.begin(); itr != kptMatches.end(); ++itr) 
    {
        prevKp = kptsPrev[itr->queryIdx].pt;
        currKp = kptsCurr[itr->trainIdx].pt;
        
        if ((currKp.x > boundingBox.roi.x) && (currKp.x < (boundingBox.roi.x + boundingBox.roi.width)) &
            (currKp.y > boundingBox.roi.y) && (currKp.y < (boundingBox.roi.y + boundingBox.roi.height)) ) 
        {        
            kptDist = cv::norm(currKp - prevKp);
            if (kptDist < meanDist) {
                boundingBox.keypoints.push_back(kptsCurr[itr->trainIdx]);
                boundingBox.kptMatches.push_back(*itr);
            }
        }
    }

    cout << "Keypoints used = " << boundingBox.keypoints.size() << endl;
}   


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    //cout << "Entering computeTTCLidar() ..." << endl;
    //cout << "Number of Prev Lidar points = " << lidarPointsPrev.size() << endl;
    //cout << "Number of Curr Lidar points = " << lidarPointsCurr.size() << endl;
    //cout << "FrameRate = " << frameRate << endl;

    double minReflectivity = 0.2;   // Filter out points from low reflectivity targets that may
                                    // unrelaiable distance measurements

    // Extract minimum and median X distances:
    std::vector<double> prevDistPoints;
    std::vector<double> currDistPoints;
    double prevMinDist = 1e9;
    double currMinDist = 1e9;
    uint i = 0;
    for (auto itr = lidarPointsPrev.begin(); itr != lidarPointsPrev.end(); ++itr) {
        if (itr->r > minReflectivity) {
            prevDistPoints.push_back(itr->x);
            prevMinDist = prevMinDist > itr->x ? itr->x : prevMinDist;
        }

        //if (i++ < 50) cout << "Prev x = " << itr->x << ", r = " << itr->r << endl;
    }
    i = 0;
    for (auto itr = lidarPointsCurr.begin(); itr != lidarPointsCurr.end(); ++itr) {
        if (itr->r > minReflectivity) {
            currDistPoints.push_back(itr->x);
            currMinDist = currMinDist > itr->x ? itr->x : currMinDist;
        }

        //if (i++ < 50) cout << "Curr x = " << itr->x << ", r = " << itr->r << endl;
    }
    // Calculate Median Distances
    double prevMedianDist = calcMedianDistance(prevDistPoints);
    double currMedianDist = calcMedianDistance(currDistPoints);

    cout << "Lidar Prev Median Dist = " << prevMedianDist << ", Curr Median Dist = " << currMedianDist << endl;
    cout << "Lidar Prev Min Dist = " << prevMinDist << ", Curr Min Dist = " << currMinDist << endl;

    double dT = 1.0 / frameRate;
    double TTCfromMedianX = prevMedianDist * dT / (prevMedianDist - currMedianDist);
    double TTCfromMinX = prevMinDist * dT / (prevMinDist - currMinDist);

    cout << "Lidar TTC from Median Values = " << TTCfromMedianX << endl;
    cout << "Lidar TTC from Min Values = " << TTCfromMinX << endl;

    TTC = TTCfromMedianX;

    cout << "TTC Lidar = " << TTC << endl;

}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{

    printBoundingBoxMatchData(matches, prevFrame, currFrame, false);
 
    int matchLimit = 0;   // limit data for debugging 

    int maxBoxID = (prevFrame.boundingBoxes.size() > currFrame.boundingBoxes.size() ? prevFrame.boundingBoxes.size() : currFrame.boundingBoxes.size());
    cout << "maxBoxID = " << maxBoxID << endl;
    std::vector<int> numMatches(maxBoxID);
    for (auto itrN = numMatches.begin(); itrN != numMatches.end(); ++itrN) {
        *itrN = 0;
    }
    std::multimap<int, int> matchCandidates;

    cout << "Number of matches = " << matches.size() << endl;
    for (auto itrM = matches.begin(); itrM != matches.end(); ++itrM) 
    {
        cv::Point prevKp;
        cv::Point currKp;
        // Get Matching Keypoint Locations
        prevKp = prevFrame.keypoints[itrM->queryIdx].pt;
        currKp = currFrame.keypoints[itrM->trainIdx].pt;
        //cout << "First point" << prevKp << endl;
        //cout << "Second point" << currKp << endl;        

        // Check if keypoints are withing Bounding Boxes
        std::vector<int> prevBoxIDs;
        std::vector<int> currBoxIDs;
        for (auto itrBB = prevFrame.boundingBoxes.begin(); itrBB != prevFrame.boundingBoxes.end(); ++itrBB)
        {
            if ((prevKp.x > itrBB->roi.x) && (prevKp.x < (itrBB->roi.x + itrBB->roi.width)) &&
                (prevKp.y > itrBB->roi.y) && (prevKp.y < (itrBB->roi.y + itrBB->roi.height)) ) {
                    prevBoxIDs.push_back(itrBB->boxID);
                    //cout << "Prev KP " << prevKp.x << "," << prevKp.y << " in BB: " << itrBB->roi.x << "," << itrBB->roi.y 
                    //     << " -> " << itrBB->roi.x + itrBB->roi.width << "," << itrBB->roi.y + itrBB->roi.height << endl;                   
            }
        }
        for (auto itrBB = currFrame.boundingBoxes.begin(); itrBB != currFrame.boundingBoxes.end(); ++itrBB)
        {
            if ((currKp.x > itrBB->roi.x) && (currKp.x < (itrBB->roi.x + itrBB->roi.width)) &&
                (currKp.y > itrBB->roi.y) && (currKp.y < (itrBB->roi.y + itrBB->roi.height)) ) {
                    currBoxIDs.push_back(itrBB->boxID);
                    //cout << "Curr KP " << currKp.x << "," << currKp.y << " in BB: " << itrBB->roi.x << "," << itrBB->roi.y 
                    //     << " -> " << itrBB->roi.x + itrBB->roi.width << "," << itrBB->roi.y + itrBB->roi.height << endl;
            }
        }

        // Add matches to candidate map
        for (auto itrP = prevBoxIDs.begin(); itrP != prevBoxIDs.end(); ++itrP) {
            for (auto itrC = currBoxIDs.begin(); itrC != currBoxIDs.end(); ++itrC) {
                matchCandidates.insert(std::make_pair(*itrP, *itrC));
            }
        }
        prevBoxIDs.clear();
        currBoxIDs.clear();

        //if (matchLimit++ > 50) break;

    }
    
    cout << "# of Match Candidates: " << matchCandidates.size() << endl;
    
    // Find best Match from match candidates
    for (int prevID = 0; prevID < maxBoxID; prevID++) {
        for (auto itrM = matchCandidates.begin(); itrM != matchCandidates.end(); ++itrM) {
            if (itrM->first == prevID) {
                numMatches[itrM->second]++;
            }
        }

        int bestID = std::distance(numMatches.begin(), std::max_element(numMatches.begin(), numMatches.end()));
        if (numMatches[bestID] > 0) {
            bbBestMatches.insert(std::make_pair(prevID, bestID));
        }

        for (auto itrN = numMatches.begin(); itrN != numMatches.end(); ++itrN) {
            *itrN = 0;
        }
    } // for prevID

}


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


double calcMedianDistance(std::vector<double> &dist)
{
    size_t size = dist.size();

    if (size == 0) {
        return 0.0;
    }
    else {
        std::sort(dist.begin(), dist.end());
        if ((size % 2) == 0) {
            return (dist[size/2 -1] + dist[size/2]) / 2;
        } else {
            return dist[size/2];
        }
    }
}


void printDataFrame(DataFrame &df, bool bVis)
{
    if (!bVis) {
        return;
    }
    //std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cout << "DataFrame: # of keypoints = " << df.keypoints.size() << endl;    
    //cv::Mat descriptors; // keypoint descriptors
    cout << "DataFrame:descriptors: " << df.descriptors.rows << " x " << df.descriptors.cols << endl;
    //std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    cout << "DataFrame: # of kptMatches = " << df.kptMatches.size() << endl;    
    //std::vector<LidarPoint> lidarPoints;
    cout << "DataFrame: # of lidarPoints = " << df.lidarPoints.size() << endl;
    //std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    cout << "DataFrame: # of boundingBoxes = " << df.boundingBoxes.size() << endl;
    //std::map<int,int> bbMatches; // bounding box matches between previous and current frame
    cout << endl;
    for (auto itr = df.boundingBoxes.begin(); itr != df.boundingBoxes.end(); ++itr)
    {
        int boxId = itr->boxID;
        cout << "    BB[" << boxId << "]: trackID = " << itr->trackID <<endl;
        cout << "    BB[" << boxId << "]: Dims = (" << itr->roi.x << "," << itr->roi.y << "),(" << itr->roi.width << "," << itr->roi.height << ")" << endl;
        cout << "    BB[" << boxId << "]: classID = " << itr->classID << endl;
        cout << "    BB[" << boxId << "]: confidence = " << itr->confidence << endl;
        //std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
        cout << "    BB[" << boxId << "]: # of lidarPoints = " << itr->lidarPoints.size() << endl;
        // std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
        cout << "    BB[" << boxId << "]: # of keypoints = " << itr->keypoints.size() << endl;
        //std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
        cout << "    BB[" << boxId << "]: # of kptMatches = " << itr->kptMatches.size() << endl;
        cout << endl;
    }
}

void printBoundingBoxData(BoundingBox &bb, bool bVis)
{
    if (bVis) {
        cout << "boxID = " <<bb.boxID <<endl;
        cout << "trackID = " << bb.trackID <<endl;
        cout << "Dims = (" << bb.roi.x << "," << bb.roi.y << "),(" << bb.roi.width << "," << bb.roi.height << ")" << endl;
        cout << "classID = " << bb.classID << endl;
        cout << "confidence = " << bb.confidence << endl;
        //std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
        cout << "# of lidarPoints = " << bb.lidarPoints.size() << endl;
        // std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
        cout << "# of keypoints = " << bb.keypoints.size() << endl;
        //std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
        cout << "# of kptMatches = " << bb.kptMatches.size() << endl;
        cout << endl;
    }
}


void printBoundingBoxMatchData(std::vector<cv::DMatch> &matches, DataFrame &prevFrame, DataFrame &currFrame, bool bVis)
{
    if (bVis) {
        cout << "printBoundingBoxMatchData() ..." << endl;
        cout << "===============================" << endl;
        cout << "Number of matches = " << matches.size() << endl;
        cout << "Previous Frame:" << endl;
        cout << "---------------" << endl;
        printDataFrame(prevFrame, true);
        cout << "Current Frame:" << endl;
        cout << "--------------" << endl;
        printDataFrame(currFrame, true);

        cout << "Number of keypoint matches = " << matches.size() << endl;
        int i = 0;
        for (auto itr = matches.begin(); itr != matches.end(); ++itr) {
            cout << "- - - - - - " << endl;
            cout << "distance = " << itr->distance << endl;
            cout << "imgIdx = " << itr->imgIdx << endl;
            cout << "queryIdx = " << itr->queryIdx << endl;
            cout << "trainIdx = " << itr->trainIdx << endl;
            cout << "First point" << prevFrame.keypoints[itr->queryIdx].pt << endl;
            cout << "Second point" << currFrame.keypoints[itr->trainIdx].pt << endl;
            if (i++ > 20) {
                break;
            }
        }

        cout << "---------------------" << endl;
        cout << "Previous frame keypoints = " << prevFrame.keypoints.size() << endl;
        i = 0;
        for (auto itr = prevFrame.keypoints.begin(); itr != prevFrame.keypoints.end(); ++itr) {
            cout << "- - - - - - " << endl;
            cout << "angle = " << itr->angle << endl;
            cout << "class_id = " << itr->class_id << endl;
            cout << "octave = " << itr->octave << endl;
            cout << "point = " << itr->pt.x << "," << itr->pt.y << endl;
            cout << "response = " << itr->response << endl;
            cout << "size = " << itr->size << endl;
            if (i++ > 10) {
                break;
            }
        }

        cout << "---------------------" << endl;
        cout << "Current frame keypoints = " << prevFrame.keypoints.size() << endl;
        i = 0;
        for (auto itr = currFrame.keypoints.begin(); itr != currFrame.keypoints.end(); ++itr) {
            cout << "- - - - - - " << endl;
            cout << "angle = " << itr->angle << endl;
            cout << "class_id = " << itr->class_id << endl;
            cout << "octave = " << itr->octave << endl;
            cout << "point = " << itr->pt.x << "," << itr->pt.y << endl;
            cout << "response = " << itr->response << endl;
            cout << "size = " << itr->size << endl;
            if (i++ > 10) {
                break;
            }
        }
    }   // bVis
}
