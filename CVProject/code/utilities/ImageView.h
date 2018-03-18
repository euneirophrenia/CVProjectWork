//
// Created by Marco DiVi on 15/03/18.
//

#ifndef PROJECTWORK_IMAGEVIEW_H
#define PROJECTWORK_IMAGEVIEW_H

#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>


class ImageWindow {

    protected:
    static void OnMouse( int event, int x, int y, int, void* ) {
        cv::Point where = cv::Point2d(x,y);

        switch (event) {
            case cv::EVENT_LBUTTONUP: {

                break;
            }

#ifdef _WIN32
            //Only windows supports mouse wheel events
            case cv::EVENT_MOUSEWHEEL: {

                return;
            }
#endif

            default: return;
        }
    }

    static void OnZoomBar(int level, void* params) {


    }

    public:
        std::string name;

    explicit ImageWindow(std::string name, cv::Mat& image) {
        this->name = name;
        this->image = cv::Mat(image);

        cv::namedWindow(name, cv::WINDOW_NORMAL);

        cv::setMouseCallback(name, OnMouse, nullptr);
        cv::createTrackbar("zoom", name, nullptr, 100, OnZoomBar);
    }

    void show() {
        cv::imshow(name, image);
        cv::waitKey(0);
    }

    private:
        cv::Mat image;

};


#endif //PROJECTWORK_IMAGEVIEW_H
