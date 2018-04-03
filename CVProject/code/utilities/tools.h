//
// Created by Marco DiVi on 04/03/18.
//


#ifndef OPENCV_UTILITIES_H
#define OPENCV_UTILITIES_H

#include <chrono>
#include <utility>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "opencv2/opencv.hpp"
#include <functional>
#include <math.h>

#define PI 3.14159265

typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

/**
 * To measure a function execution time.
 * Credits at StackOverflow (https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c/33900479#33900479).
 * @tparam F
 * @tparam Args
 * @param func
 * @param args
 * @return
 */
template<typename F, typename... Args>
double funcTime(F func, Args&&... args){
    TimeVar t1=timeNow();
    func(std::forward<Args>(args)...);
    return duration(timeNow()-t1);
}

/**
 * Function to delete indices from a vector, returns a new vector.
 * @tparam T
 * @param data from which to delete
 * @param indicesToDelete
 * @return a new vector properly shrunk.
 */
//todo: optimize! i'm sure something can be done

template<typename T>
inline std::vector<T> erase_indices(const std::vector<T>& data, std::vector<size_t>& indicesToDelete/* can't assume copy elision, don't pass-by-value */)
{
    if(indicesToDelete.empty())
        return data;

    if (indicesToDelete.size() > data.size()){
        throw std::invalid_argument("Too many indices to delete.");
    }

    std::vector<T> ret;

    ret.reserve(data.size() - indicesToDelete.size());

    bool mask[data.size()];
    for (int i=0; i< data.size(); i++) {
        mask[i] = true;
    }
    for (auto i : indicesToDelete) {
        mask[i] = false;
    }

    for (int i=0; i< data.size();i++) {
        if (mask[i])
            ret.push_back(data[i]);
    }

    return ret;
}

//TODO: optimize this shit ASAP
cv::Mat erase_rows(cv::Mat& data, std::vector<size_t>& indices) {

    if(indices.empty())
        return data;

    if (indices.size() > data.rows){
        throw std::invalid_argument("Too many indices to delete.");
    }
    cv::Mat res = cv::Mat::zeros(CvSize(data.cols, data.rows - indices.size()), data.type());

    bool mask[data.rows];
    for (int i=0; i< data.rows; i++) {
        mask[i] = true;
    }
    for (auto i : indices) {
        mask[i] = false;
    }

    int currentRow = 0;
    for (int i=0; i< data.rows;i++) {
        if (mask[i]) {
            for (int j = 0; j < data.cols * data.channels(); j++) {
                res.data[data.cols * data.channels() * currentRow + j] = data.data[data.cols * data.channels() * i + j];
            }
            currentRow++;
        }
    }

    return res;
}

char readUntil(std::istream* in, char delim, std::function<void(char)> callback = [](char c) { }) {
    char current = -1;
    char last = current;
    while (!in->eof() && current != delim) {
        last = current;
        in->read(&current, 1);
        callback(current);
    }
    return last;
}

void readLine(std::istream* in, std::string* line) {
    readUntil(in, '\n', [=](char c) {
        *line+=c;
    });
}

std::string sanifyJSON(std::string filename) {
    std::ifstream inputstream;
    std::string res, line;
    inputstream.open(filename);
    char c;
    while (!inputstream.eof()) {
        inputstream.read(&c, 1);
        line+=c;
        if (c=='\n') {
            auto match = line.find("//");
            if (match != std::string::npos) {
                line.erase(line.begin() + match, line.end());
            }
            res += line;
            line = "";
        }
    }
    res+=c;

    inputstream.close();
    return res;
}


cv::Vec2d rotate(cv::Vec2d src, double angle) {

    auto res = cv::Vec2d(src);

    double radians = angle * PI/ 180;

    res[0] = src[0] * cos(radians) - src[1] * sin(radians);
    res[1] = src[0] * sin(radians) + src[1] * cos(radians);

    return res;
}

cv::Mat diffuse(cv::Mat in, int side,  double sigmax, double sigmay, double gain=1.0f) {
    cv::Mat res(in.rows, in.cols, in.type());

    cv::GaussianBlur(in, res, CvSize(side, side), sigmax, sigmay);

    return res + gain*in;

}

/**
 * Algorithm to extract the skeleton of an image.
 * Credits http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
 * @param src the input image
 * @return the skeleton image
 */
cv::Mat skeleton(cv::Mat src) {

    cv::Mat img;
    cv::threshold(src, img, 127, 255, cv::THRESH_BINARY);
    cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;

    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    bool done;
    do {
        cv::erode(img, eroded, element);
        cv::dilate(eroded, temp, element); // temp = open(img)
        cv::subtract(img, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(img);

        done = (cv::countNonZero(img) == 0);
    } while (!done);

    return skel;

}

template<typename T>
int indexOf(T elem, std::vector<T> vec) {
    for (int i=0; i< vec.size(); i++) {
        if (vec[i] == elem)
            return i;
    }

    return -1;
}

double distance(cv::Point a, cv::Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}


std::string fileName(const std::string& str, bool withoutExtension = false)
{
    size_t found = str.find_last_of("/\\");
    std::string path = str.substr(found+1); // check that is OK

    if (withoutExtension) {
        found = path.find_last_of(".");
        if (found > 0)
            path = path.substr(0, found);
    }

    return path;
}
#endif //OPENCV_UTILITIES_H
