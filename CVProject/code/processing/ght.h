//
// Created by Marco DiVi on 08/03/18.
//

#ifndef PROJECTWORK_GHT_H
#define PROJECTWORK_GHT_H
#include <vector>
#include "opencv2/opencv.hpp"

struct  HoughModel {
private:
    std::vector<cv::Vec2d> model;

public:
    cv::Vec2d operator[](int i) {
        return model[i];
    }

    explicit HoughModel(std::vector<cv::Vec2d> m){
        this->model = std::vector<cv::Vec2d>(m);
    }

};

#endif //PROJECTWORK_GHT_H
