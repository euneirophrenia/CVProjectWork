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

struct RichImage {
    std::string path;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat features;
    std::vector<cv::Vec2d> houghModel;

    private:
        int _scale = -1;

        void buildHoughModel() {
            ///find barycenter
            cv::Point2f bary;
            for (auto kp : this->keypoints) {
                bary += kp.pt;
            }
            bary.x /= this->keypoints.size();
            bary.y /= this->keypoints.size();

            for (auto kp : this -> keypoints){
                this->houghModel.push_back(cv::Vec2d(bary.x - kp.pt.x, bary.y - kp.pt.y) / kp.size);
            }
        }

    public:

        void build(Algorithm* algo, bool andBuildHough = false){

            algo->detector->detectAndCompute(this->image, cv::Mat(), this->keypoints, this->features);

            /// root sift, comment out if not using sift/surf maybe
            for(int i=0; i< features.rows; ++i) {

                features.row(i) = features.row(i) / cv::sum(features.row(i))[0];
                cv::sqrt(features.row(i), features.row(i));
            }
            if (andBuildHough)
                buildHoughModel();

        }

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

        explicit RichImage(std::string path, int mode = cv::IMREAD_GRAYSCALE) {
            this->path = path;
            this->image = cv::imread(path, mode);
        }

        /// backwards compatibility, I initially didn't account for
        explicit RichImage() {}

        ///return the lenght (more or less) of the longest vertical line
        // it is a (very) rough estimation of the size of the objects in the scene, it's a quantity proportional to that
        int approximateScale() {
            if (_scale < 0) {
                cv::Mat edges;

                cv::Canny(this->image, edges, 50, 150);
                std::vector<cv::Vec4i> lines;
                cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 2, 5);

                _scale = 0;
                for (size_t i = 0; i < lines.size(); i++) {
                    cv::Vec4i l = lines[i];
                    if (l[0] == l[2]) {
                        if (abs(l[1] - l[3]) > _scale)
                            _scale = abs(l[1] - l[3]);
                    }
                }
            }

            return 2*_scale;
        }


};



/**
 * Global entry point for the images. not the smartest idea but hey.
 */
ResourcePool<std::string, RichImage> Images;


std::function<RichImage*(std::string)> load(int flag) {
    return [=](std::string filename) {
        RichImage* res = new RichImage();
        res->image = cv::imread(filename, flag);
        if (res->image.empty()){
            throw std::invalid_argument("Image not loaded properly " + filename);
        }
        res->path = filename;
        return res;

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
