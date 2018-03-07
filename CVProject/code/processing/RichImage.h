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

#include "opencv2/xfeatures2d/nonfree.hpp"

const Context& context = Context::getInstance();
/**
 * Global entry point for caching images, read them only once and have them forever.
 * Might not be a good idea, but it could be used to ensure that every image is loaded in the same way or preprocessed
 * properly. Its use is not mandatory. I think i will expand it by caching "Rich" images.
 * I.e images with also already computed features and keypoints so that we don't need to repeat that across various scenes.
 */

struct RichImage {
    std::string path;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat features;

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
};

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


#endif //TOOLS_H
