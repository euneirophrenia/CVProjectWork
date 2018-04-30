//
// Created by Marco DiVi on 13/03/18.
//

#ifndef PROJECTWORK_BLOB_H
#define PROJECTWORK_BLOB_H

#include "opencv2/opencv.hpp"

struct Blob {

    public:
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

        inline Blob operator += (Blob& other) {
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

        inline bool isInside(cv::Mat image) {
            return position.x >= 0 && position.y >= 0 && position.x < image.cols && position.y < image.rows;
        }

        inline operator std::string() {
			std::stringstream ss;
			ss << model->path << " @" << position << " ("<< confidence << ")";
			return ss.str();
        }

};

struct BlobProxy {
    private:
        std::vector<Blob> queue;

    public:
        cv::Point2d avgPosition;

        void add(Blob& b) {
            if (queue.size() == 0) {
                avgPosition = b.position;
                return;
            }
            avgPosition *= (float)queue.size();
            avgPosition += b.position;
            queue.push_back(b);
            avgPosition /=  (float)queue.size();
        }

        Blob toBlob() {
            if (queue.size() == 0)
                throw std::invalid_argument("Empty proxy can't be turned to blob (yet).");

            Blob b = queue[0];

            for (int i=1; i< queue.size(); i++)
                b += queue[i];

            return b;
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
