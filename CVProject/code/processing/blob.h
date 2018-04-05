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

        this->position = (this->confidence*this->position + other.confidence*other.position)/(this->confidence+other.confidence);
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

struct VotingMatrix {
    private:
        std::map<RichImage*, std::vector<Blob>> blobs;
        RichImage* scene;

    public:
        explicit VotingMatrix(RichImage* scene) {
            this->scene = scene;
        }

        void castVote(cv::DMatch match, RichImage *forModel) {
            double scale = scene->keypoints[match.trainIdx].size;
            double angle = scene->keypoints[match.trainIdx].angle - forModel->keypoints[match.queryIdx].angle;

            cv::Point2d scenept = scene->keypoints[match.trainIdx].pt;
            cv::Point2d estimated_bary;
            cv::Vec2d houghmodel = forModel->houghModel[match.queryIdx];


            houghmodel = rotate(houghmodel, angle);

            estimated_bary.x = (scenept.x +  scale * houghmodel[0]);
            estimated_bary.y = (scenept.y +  scale * houghmodel[1]);

            Blob b;
            b.position = estimated_bary;
            b.confidence = 1;
            b.matches.push_back(match);
            b.area = 1;
            b.model = forModel;

            blobs[forModel].push_back(b);

        }

        void collapse(double collapsingDistance) {
            for (auto pair : blobs) {
                cv::Mat votes = cv::Mat::zeros(scene->image.size(), CV_32F);
                for (auto blob : pair.second) {

                    if (blob.position.x < collapsingDistance/2 || blob.position.y < collapsingDistance/2 ||
                            blob.position.x > votes.cols - collapsingDistance/2 || blob.position.y > votes.rows - collapsingDistance/2) {
                        std::cerr << "[IGNORING] " << blob << "\n";
                        continue;
                    }

                    for (int x = int(blob.position.x - collapsingDistance/2) ; x < blob.position.x + collapsingDistance/2 ; x++){
                        for (int y = int(blob.position.y - collapsingDistance/2) ; y < blob.position.y + collapsingDistance/2 ; y++){
                            double dist = cv::norm(blob.position - cv::Point2d(y,x));
                            votes.at<float>(y,x) += 1;
                        }
                    }
                }
                cv::Mat labels(scene->image.size(), CV_32S);
                int howmany = cv::connectedComponents(votes > 0, labels);
                if (howmany < 2) {
                    continue;
                }

                /*for (int i1 = 0; i1< labels.rows; i1++){
                    for (int j1=0; j1< labels.cols; j1++) {
                        int lab = labels.at<int>(i1,j1);
                        if (lab == 0)
                            std::cout << " - ";
                        else
                            std::cout << " " << lab << " ";
                    }
                    std::cout << "\n";
                }

                exit(0);*/

                std::vector<Blob> compacted;
                compacted.reserve( howmany - 1);
                for (int w = 0; w < howmany-1; w++) {
                    Blob prototype;
                    prototype.model =  pair.first;
                    prototype.confidence = 0;
                    prototype.area = 0;
                    prototype.position=cv::Point2d(0,0);
                    prototype.matches = std::vector<cv::DMatch>();
                    compacted.push_back(prototype);
                }

                for (auto b : pair.second) {
                    if (scene->contains(b.position)) {
                        int label = labels.at<int>(b.position);
                        if (label == 0) {
                            continue;
                        }
                        compacted[label - 1] += b;
                    }
                }

                blobs[pair.first] = compacted;

            }
        }

        void prune(double pruneDistance) {
            std::vector<Blob> allblobs;
            std::vector<size_t> indicesToRemove;

            for (auto pair : blobs) {
                auto matches = pair.second;

                for (auto blob : matches) {
                    indicesToRemove.clear();
                    bool best=true;

                    for (int k =0; k<allblobs.size() && best; k++){
                        double dist = distance(allblobs[k].position, blob.position);
                        if (allblobs[k].confidence >= blob.confidence &&  dist <= pruneDistance) {
                            best = false;
                        }

                        if (allblobs[k].confidence < blob.confidence && dist <= pruneDistance) {
                            indicesToRemove.push_back(k);
                            if (blob.model == allblobs[k].model) {
                                continue;
                            }
                            else {
#ifdef DEBUG
                                std::cerr << "[PRUNING] " << allblobs[k].model ->path << " (" << allblobs[k].position << ", " <<
                                          allblobs[k].confidence << ") in favor of " << blob.model->path << " (" << blob.position
                                          << ", " <<
                                          blob.confidence << " ) - " << dist << "\n";
#endif
                            }
                        }
                    }

                    if (best)
                        allblobs.push_back(blob);

                    allblobs = erase_indices(allblobs, indicesToRemove);
                }

            }

            blobs.clear();
            for (auto b : allblobs) {
                blobs[b.model].push_back(b);
            }
        }

        ///filter out all blobs with confidence < threhsold * best_confidence within the same model
        void relativeFilter(double threshold = 0.5) {
            std::vector<size_t> indicesToRemove;
            for (auto pair:blobs) {

                double best=0;
                for (auto blob : pair.second) {
                    if (blob.confidence > best)
                        best = blob.confidence;
                }

                for (size_t i=0; i < pair.second.size(); i++) {
                    if (pair.second[i].confidence < threshold * best)
                        indicesToRemove.push_back(i);
                }

                blobs[pair.first] = erase_indices(blobs[pair.first], indicesToRemove);
                indicesToRemove.clear();
            }

        }

        ///filter out all blobs with less than an absolute value for confidence
        void absoluteFilter(double threshold = context["MIN_HOUGH_VOTES"]) {
            std::vector<size_t> indicesToRemove;
            for (auto pair:blobs) {

                for (size_t i=0; i < pair.second.size(); i++) {
                    if (pair.second[i].confidence < threshold)
                        indicesToRemove.push_back(i);
                }

                blobs[pair.first] = erase_indices(blobs[pair.first], indicesToRemove);
                indicesToRemove.clear();
            }

        }

        std::vector<Blob> operator[] (RichImage* model) {
            return blobs[model];
        }

        std::map<RichImage*, std::vector<Blob>> asMap() {
            return this->blobs;
        }
};

std::vector<cv::KeyPoint> simpleBlob(cv::Mat& in) {
    auto detector = cv::SimpleBlobDetector::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(in, keypoints);
    return keypoints;
}






#endif //PROJECTWORK_BLOB_H
