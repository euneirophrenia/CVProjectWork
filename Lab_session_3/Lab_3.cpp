//Comment the following line to use the C interface, uncomment to use the c++ interface
#define OPENCV_CPP_INTERFACE

#ifndef OPENCV_CPP_INTERFACE
#include "opencv/cv.h"
#include "opencv/highgui.h"
#else
#include <iostream>
#include "opencv2/opencv.hpp"
#endif

#include <cstdio>
#include <math.h>

#ifndef OPENCV_CPP_INTERFACE
//WORK HERE FOR THE C INTERFACE

void convolution(IplImage * input, IplImage * output, float * kernel, int kernel_side){
	//=====================EXERCISE 1==========================
	//apply the convolutional operation on each pixel of input using the passed kernel, save the result in output
	return;
}

void TwoFrameDifference(IplImage*current_frame, IplImage*previous_frame, IplImage * output, int threshold){
	//=====================EXERCISE 2==========================
	//Compute two frame difference between current and previous, save the difference mask in output
	return;
}

int main(int argc, char**argv){
	//====================Exercise 0.a========================:
	//Open a video stream either from file or from your webcam
	//=======================================================

	//create placeholder for original and modified frame
	IplImage * previous_frame = NULL;
	IplImage * current_frame = NULL;
	IplImage * frame_converted = NULL;
	
	int kernel_side = 3;
	//uncomment only one filter at a time

	//edge detector filter
	//float kernel_filter[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };

	//gaussian blur filter
	//float kernel_filter[9] = { 1.0f / 16, 2.0f / 16, 1.0f / 16, 2.0f / 16, 4.0f / 16, 2.0f / 16, 1.0f / 16, 2.0f / 16, 1.0f / 16 };
	
	//mean filter
	float kernel_filter[9] = { 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9 };

	//====================Exercise 0.b========================
	//write a while loop to iterate over all the frame in the video sequence, store each frame in current_frame
	//=======================================================
	while (){
		
		//=============================EXERCISE 1===============================
		//implement convolution function and try out different kernels on current_frame

		//convolution(current_frame,frame_converted,kernel_filter,kernel_side);

		//=====================================================================


		//=============================EXERCISE 2===============================
		//Implement two frame difference function and try it out on a video sequence

		//TwoFrameDifference(current_frame,previous_frame, frame_converted, 30);

		//when the difference mask is computed at the end of the iteration copy current frame in previous frame fro teh next iteration
		//======================================================================

		//Visualize orignal frame 
		cvShowImage("Original", current_frame);
		//uncomment the following line to display frame_converted
		//cvShowImage("Elaborated", frame_converted);
		cvWaitKey(1);

		
	}

	//====================Exercise 0.c========================
	//when done close the video stream
	//=======================================================

	cvReleaseImage(&previous_frame);
	cvReleaseImage(&current_frame);
	cvReleaseImage(&frame_converted);
}

#else
//WORK HERE FOR THE C++ INTERFACE

void convolution(const cv::Mat & input, cv::Mat & output, const std::vector<float> & kernel, const int kernel_side){
	//=====================EXERCISE 1==========================
	//apply the convolutional operation on each pixel of input using the passed kernel, save the result in output
	return;
}

void two_frame_difference(const cv::Mat & current, const cv::Mat & previous, cv::Mat & output, const int threshold){
	//=====================EXERCISE 2==========================
	//Compute two frame difference between current and previous, save the difference mask in output
	return;
}

int main(int argc, char**argv){
	//====================Exercise 0.a========================:
	//Open a video stream either from file or from your webcam
	//=======================================================

	//create placeholder for original and modified frame
	cv::Mat previous_frame;
	cv::Mat current_frame;
	cv::Mat frame_converted;

	int kernel_side = 3;
	std::vector<float> kernel_filter(kernel_side*kernel_side);
	//uncomment only one filter at a time
	
	//edge detector filter
	//float k[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
	
	//gaussian blur filter
	//float k[9] = { 1.0f / 16, 2.0f / 16, 1.0f / 16, 2.0f / 16, 4.0f / 16, 2.0f / 16, 1.0f / 16, 2.0f / 16, 1.0f / 16 };
	
	//mean filter
	float k[9] = { 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9 };
	kernel_filter.assign(k,k+9);

	//====================Exercise 0.b========================
	//write a while loop to iterate over all the frame in the video sequence, store each frame in current_frame
	//=======================================================
	while (){

		
		//=============================EXERCISE 1===============================
		//implement convolution function and try out different kernels on current_frame

		//convolution(current_frame,frame_converted,kernel_filter,kernel_side);

		//=====================================================================


		//=============================EXERCISE 2===============================
		//Implement two frame difference function and try it out on a video sequence

		//TwoFrameDifference(current_frame,previous_frame, frame_converted, 30);

		//when the difference mask is computed at the end of the iteration copy current frame in previous frame fro teh next iteration
		//======================================================================


		//Visualize orignal frame 
		cv::imshow("Original", current_frame);
		//uncomment the following line to display frame_converted
		//cv::imshow("Modified", frame_converted);
		cv::waitKey(1);

		

	}
	
	//====================Exercise 0.c========================
	//when done close the video stream
	//=======================================================

	std::cout << "All frames in the video elaborated..." << std::endl;
	std::cout << "Ending execution" << std::endl;
	return 0;
}
#endif