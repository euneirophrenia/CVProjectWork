//
// Created by Marco DiVi on 03/03/18.
//

/**
 * This module collects some useful / recurrent functions I needed throughout the project.
 */

//TODO:: test the strictDescription, optimize it and complete translation of the python work

#ifndef TOOLS_H
#define TOOLS_H

#include "../utilities/ResourcePool.h"
#include "../utilities/tools.h"
#include "../context.h"
#include "ght.h"

#include "opencv2/xfeatures2d/nonfree.hpp"

const Context& context = Context::getInstance();

struct RichImage {
    std::string path;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat features;
    HoughModel* houghModel;

    public:
        operator cv::Mat() const {
            return image;
        }

        operator cv::InputArray () const {
            return cv::InputArray(image);
        }

        void show(std::string windowname, int waitkey=0) {
            cv::imshow(windowname, image);
            cv::waitKey(waitkey);
        }

        explicit RichImage(std::string path, cv::Feature2D* detector, int mode = cv::IMREAD_GRAYSCALE) {
            this->path = path;
            this->image = cv::imread(path, mode);
            detector->detectAndCompute(this->image, cv::Mat(), this->keypoints, this->features);
        }

        /// backwards compatibility, I initially didn't account for
        explicit RichImage() {}

        void buildHoughModel() {
            std::vector<cv::Vec2d> res;
            ///find barycenter
            cv::Point2f bary;
            for (auto kp : this->keypoints)
                bary += kp.pt / kp.size;
            bary.x /= this->keypoints.size();
            bary.y /= this->keypoints.size();

            for (auto kp : this->keypoints) {
                res.push_back(cv::Vec2d(kp.pt.x - bary.x, kp.pt.y - bary.y));
            }
            this->houghModel = new HoughModel(res);
    }

};



/**
 * Global entry point for the images. not the smartest idea but hey.
 */
ResourcePool<std::string, RichImage> Images;


std::function<RichImage(std::string)> load(int flag) {
    return [=](std::string filename) {
        RichImage res;
        res.image = cv::imread(filename, flag);
        if (res.image.empty()){
            throw std::invalid_argument("Image not loaded properly " + filename);
        }
        res.path = filename;
        return res;

    };
}

std::function<RichImage(std::string)> init(cv::Ptr<cv::Feature2D> detector, int flag) {
    return [&detector, flag](std::string filename) {
        RichImage* res = new RichImage(filename, detector, flag);

        if (res->image.empty()){
            throw std::invalid_argument("Image not loaded properly " + filename);
        }
        return *res;
    };
}

int sizeOfImage(RichImage* image) {
    int res = sizeof(image->path);
    for (auto x : image->keypoints)
        res+=sizeof(x);
    res+=sizeof(image->features);
    res+=sizeof(image->image);

    return res;
}


#endif //TOOLS_H
