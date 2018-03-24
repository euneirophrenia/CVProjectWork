//
// Created by Marco DiVi on 13/03/18.
//

#ifndef PROJECTWORK_BLOB_H
#define PROJECTWORK_BLOB_H

#include "opencv2/opencv.hpp"

struct BlobPosition {

    cv::Point2d position;
    int confidence;

};



std::vector<cv::KeyPoint> simpleBlob(cv::Mat& in) {
    auto detector = cv::SimpleBlobDetector::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(in, keypoints);
    return keypoints;
}

/*std::vector<BlobPosition> aggregate(cv::Mat& in) {

    cv::Mat temp(in.rows, in.cols, in.type());
    std::vector<BlobPosition> res;

    std::vector<cv::Point2d> potential;

    for (int i=0; i<in.rows; i++) {
        for (int j=0; j<in.cols; j++) {
            if (in.data[i*in.step + j] > 0) {
                cv::Point2d p(j,i);
                potential.push_back(p);
            }
        }
    }



    return res;

}*/






#endif //PROJECTWORK_BLOB_H
