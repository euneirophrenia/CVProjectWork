//
// Created by Marco DiVi on 13/03/18.
//

#ifndef PROJECTWORK_BLOB_H
#define PROJECTWORK_BLOB_H

#include "opencv2/opencv.hpp"

struct Blob {

    cv::Point2d position;
    float confidence;
    int area;
    std::vector<cv::DMatch> matches;
    RichImage* model;

    Blob() : position(cv::Point2d(0,0)), confidence(0.0f), area(0), model(nullptr), matches(std::vector<cv::DMatch>()) {}

    /*bool operator== (Blob& other) {
        return this->position == other.position
               && this-> area == other.area
               && this-> confidence == other.confidence
               && this->modelName == other.modelName;

    }

    bool operator != (Blob& other) {
        return ! (*this == other);
    }*/

    Blob operator += (Blob& other) {
        if (other.model != this->model) {
            return *this;
        }

        this->position = ((confidence * position) + (other.confidence * other.position))/(confidence+other.confidence);
        this->area += other.area;
        this->confidence += other.confidence;

        for (auto dm : other.matches) {
            this->matches.push_back(dm);
        }

        return *this;
    }

};

std::ostream& operator<<(std::ostream &strm, const Blob &b) {
    return strm << b.model->path << " @" << b.position << " ("<< b.confidence << ")";
}

std::vector<Blob> aggregate(cv::Mat& votes, RichImage* model) {

    cv::Mat labels(votes.size(), CV_32S);
    int howmany = cv::connectedComponents(votes > 0, labels, 8);

    if (howmany < 2)
        return std::vector<Blob>();

    int current;
    Blob blobs[howmany-1];

    for (int k=0; k< howmany-1; k++)
        blobs[k].model = model;


    for (int x= 0; x< labels.rows; x++) {
        for (int y= 0; y< labels.cols; y++) {
            current = labels.at<int>(x,y);

            ///background
            if (current == 0)
                continue;

            blobs[current-1].position += votes.at<float>(x,y) * cv::Point2d(y,x);
            blobs[current-1].confidence += votes.at<float>(x,y);
            blobs[current-1].area += 1;
        }
    }

    std::vector<Blob> actual;

    for (int i=0; i<howmany-1; i++) {
        if (blobs[i].confidence >= context["MIN_HOUGH_VOTES"]) {
            blobs[i].position /= blobs[i].confidence;
            //blobs[i].confidence /= blobs[i].area; //without this its more precise in doubtful areas

            actual.push_back(blobs[i]);
            //std::cout << blobs[i].position << ", " << blobs[i].confidence << "\n";
        }
    }

    return actual;

}





#endif //PROJECTWORK_BLOB_H
