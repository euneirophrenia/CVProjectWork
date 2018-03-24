//
// Created by Marco DiVi on 13/03/18.
//

#ifndef PROJECTWORK_BLOB_H
#define PROJECTWORK_BLOB_H

#include "opencv2/opencv.hpp"

struct BlobPosition {

    cv::Point2d position;
    float confidence;
    int area;

    BlobPosition() : position(cv::Point2d(0,0)), confidence(0.0f), area(0) {}

};



std::vector<cv::KeyPoint> simpleBlob(cv::Mat& in) {
    auto detector = cv::SimpleBlobDetector::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(in, keypoints);
    return keypoints;
}

std::vector<BlobPosition> aggregate(cv::Mat& in) {

    cv::Mat labels(in.size(), CV_32S);
    int howmany = cv::connectedComponents(in > 0, labels, 8);

    if (howmany < 2)
        return std::vector<BlobPosition>();


    int current;
    BlobPosition blobs[howmany-1];

    for (int x= 0; x< labels.rows; x++) {
        for (int y= 0; y< labels.cols; y++) {
            current = labels.at<int>(x,y);

            ///background
            if (current == 0)
                continue;

            blobs[current-1].position += in.at<int>(x,y)*cv::Point2d(y,x);
            blobs[current-1].confidence += in.at<int>(x,y);
            blobs[current-1].area += 1;
        }
    }

    std::vector<BlobPosition> actual;

    for (int i=0; i<howmany-1; i++) {
        blobs[i].position /= blobs[i].confidence;
        blobs[i].confidence /= blobs[i].area;

        actual.push_back(blobs[i]);
    }

    return actual;

}






#endif //PROJECTWORK_BLOB_H
