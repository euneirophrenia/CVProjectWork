//
// Created by Marco DiVi on 13/03/18.
//

#ifndef PROJECTWORK_BLOB_H
#define PROJECTWORK_BLOB_H

#include "opencv2/opencv.hpp"

std::vector<cv::KeyPoint> simpleBlob(cv::Mat in) {
    auto detector = cv::SimpleBlobDetector::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(in, keypoints);
    return keypoints;
}





#endif //PROJECTWORK_BLOB_H
