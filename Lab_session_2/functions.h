//Comment the following line to use the C interface, uncomment to use the c++ interface
#define OPENCV_CPP_INTERFACE

#ifndef OPENCV_CPP_INTERFACE
#include "opencv/cv.h"
#include "opencv/highgui.h"
#else
#include "opencv2/opencv.hpp"
#endif

#include <cstdio>
#include <math.h>

#ifndef OPENCV_CPP_INTERFACE
//C interface function definition
void buildHistogram(IplImage * input,unsigned int * histogram);
void resetHistogram(unsigned int * histogram);
void visualizeHistogram(unsigned int * histogram);
void linearContrastStretch(IplImage * input, IplImage * output);
void gammaCorrection(IplImage * input, IplImage * output, double r);
void imageEqualization(IplImage * input, IplImage * output, unsigned int*histogram);
void convolution(IplImage * input, IplImage * output, float * kernel, int kernel_side);
#else
//C++ interface funciton definition
void buildHistogram(const cv::Mat & input, std::vector<unsigned int> & histogram);
void visualizeHistogram(const std::vector<unsigned int> & histogram);
void linearContrastStretch(const cv::Mat & input, cv::Mat & output);
void gammaCorrection(const cv::Mat & input, cv::Mat & output, const double r);
void imageEqualization(const cv::Mat & input, cv::Mat & output, std::vector<unsigned int> & histogram);
void convolution(const cv::Mat & input, cv::Mat & output, std::vector<float> & kernel, int kernel_side);
#endif