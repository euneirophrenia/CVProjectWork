//
// Created by Marco DiVi on 03/03/18.
//

/**
 * This module collects some useful / recurrent functions I needed throughout the project.
 */

#ifndef TOOLS_H
#define TOOLS_H

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
    bool isHard = false;

    private:
        int _scale = -1;
        cv::Rect image_rect;

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

        inline void build(Algorithm* algo, bool andBuildHough = false){

            algo->detector->detectAndCompute(this->image, cv::Mat(), this->keypoints, this->features);

            /// root sift, comment out if not using sift/surf maybe
            if (!context.cli_options->at("-sift")) {
                for (int i = 0; i < features.rows; ++i) {

                    features.row(i) = features.row(i) / cv::sum(features.row(i))[0];
                    cv::sqrt(features.row(i), features.row(i));
                }
            }

            if (andBuildHough)
                buildHoughModel();

        }

        inline operator cv::Mat() const {
            return image;
        }

       inline  operator cv::InputArray () const {
            return cv::InputArray(image);
        }

        void show(std::string windowname, int waitkey=0) {
            cv::imshow(windowname, image);
            cv::waitKey(waitkey);
        }

        explicit RichImage(std::string path, int mode = cv::IMREAD_GRAYSCALE) {
            this->path = path;
            this->image = cv::imread(path, mode);
            if (this->image.empty())
                throw std::invalid_argument("Image at " + path + " not found.");
            this->image_rect = cv::Rect(cv::Point(0,0), image.size());
        }

        /// backwards compatibility, I initially didn't account for
        explicit RichImage() {}

        inline void GaussianBlur(cv::Size kernelSize = context.GAUSSIAN_KERNEL_SIZE, float xsigma = context.GAUSSIAN_X_SIGMA, float ysigma = context.GAUSSIAN_Y_SIGMA) {
            cv::Mat rescaled;
            cv::GaussianBlur(this->image, rescaled, kernelSize, xsigma, ysigma);
            this->image = rescaled;
        }

        inline void deBlur(bool fast=false) {

            if (fast) {
                cv::Mat temp;
                cv::GaussianBlur(image, temp, cv::Size(0, 0), 3);
                cv::addWeighted(image, 1.5, temp, -0.5, 0, image);
                return;
            }

            cv::Mat blurred;
            double sigma = 1, threshold = 5, amount = 1;
            cv::GaussianBlur(image, blurred, cv::Size(), sigma, sigma);
            cv::Mat lowConstrastMask = cv::abs(image - blurred) < threshold;
            cv::Mat sharpened = image*(1+amount) + blurred*(-amount);
            image.copyTo(sharpened, lowConstrastMask);
            image = sharpened;
        }

        inline void sharpen() {
            std::vector<float> kernel({-1, -1, -1, -1, 9, -1, -1, -1, -1});
            cv::filter2D(image, image, image.depth(), kernel);
        }

        inline bool contains(cv::Point p) {
            return image_rect.contains(p);
        }

        ///return the lenght (more or less) of the longest vertical line
        // it is a (very) rough estimation of the size of the objects in the scene, it's a quantity proportional to that
        inline int approximateScale() {
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

            return MIN( _scale, image.size().height);
        }

        inline void hsv(std::vector<cv::Mat> hsv) {
            cv::Mat tmp;
            image.convertTo(tmp, cv::COLOR_RGB2HSV);
            cv::split(tmp, hsv);

        }


};

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
