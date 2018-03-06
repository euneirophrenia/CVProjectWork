#define OPENCV_CPP_INTERFACE

#ifndef OPENCV_CPP_INTERFACE
	#include "opencv/cv.h"
	#include "opencv/highgui.h"
#else
	#include "opencv2/opencv.hpp"
#endif

int main(int argc, char**argv){
#ifndef OPENCV_CPP_INTERFACE
	//Declare a pointer to IplImage to handle the original image and the inverted one
	IplImage* source_image=0;
	IplImage* inverted_image = 0;

	//filepath to the image that we want to load
	char*image_filepath="../data/clown.jpg";

	//========================TO DO:EXERCISE 1=========================
	//Load in source_image the image located at image_filepath

	//=================================================================

	//Check if the image is properly laoded
	if (source_image == NULL){
		printf("Unable to load image with file path: %s\n",image_filepath);
		printf( "Ending execution..." );
		return 0;
	}

	//Print some image information
	printf("Image %s succesfully loaded\n",image_filepath);
	printf("Dimension: %dX%d\n",source_image->width, source_image->height);
	printf("Number of channels: %d\n", source_image->nChannels);
	printf("Step: %d\n", source_image->widthStep);
	printf("==========================================\n");

	//========================TO DO:EXERCISE 1=========================
	//show the image in a window named "source image"
	
	//=================================================================

	//stop the execution and wait for any key pressed by the user to resume, listen only to key pressed with focus on the GUI
	printf("Press any key while focusing on the GUI to continue...\n");
	cvWaitKey();

	//=======================TO DO: Exercise 2=========================
	//create an output iplimage with the same size of source_image and save the pointer in inverted_image
	//hint: cvGetSize(IplImage*a) to get the size of the iplimage a
	//hint: default depth for image 'IPL_DEPTH_8U'

	//=================================================================

	if (inverted_image == NULL){
		printf("Something went wrong during the creation of inverted_image, please check your code!\n");
		cvWaitKey();
		return 0;
	}
	else{
		//Print some image information
		printf("Inverted image succesfully created\n");
		printf("Dimension: %dX%d\n", inverted_image->width, inverted_image->height);
		printf("Number of channels: %d\n", inverted_image->nChannels);
		printf("Step: %d\n", inverted_image->widthStep);
		printf("==========================================\n");
	}

	//iterate over all the row and cols of source_image
	int row, col;
	int index;
	for (row = 0; row < source_image->height; row++){
		for (col = 0; col < source_image->width; col++){
			//=======================TO DO: Exercise 2=========================
			//Assign to each pixel of inverted_image 255-the corresponding pixel in source_image.
			
			//=================================================================
		}
	}

	//========================TO DO:EXERCISE 2=========================
	//show the inverted image in a window named "inverted image"

	//=================================================================

	//stop the execution and wait for any key pressed by the user to resume, listen only to key pressed with focus on the GUI
	printf("Press any key while focusing on the GUI to continue...\n");
	cvWaitKey();

	//========================TO DO:EXERCISE 3=========================
	//Modify all the code above to load a color image and invert all the three channels
	//=================================================================

	printf("Execution Completed\n");

	//release memory allocated to IplImage
	cvReleaseImage(&source_image);
	cvReleaseImage(&inverted_image);

	//destroy opencv GUI window
	cvDestroyAllWindows();

	return 0;
#else
	//Declare a mat to handle the original image
	cv::Mat source_image;
	cv::Mat inverted_image;

	//filepath to the image that we want to load
	std::string image_filepath = "../data/clown.jpg";

	//========================TO DO:EXERCISE 1=========================
	//Load in source_image the image located at image_filepath

	//================================================================

	//Check if the image is properly laoded
	if (source_image.data == NULL){
		std::cout << "Unable to load image with file path: " << image_filepath << std::endl;
		std::cout << "Ending execution..." << std::endl;
		return 0;
	}

	//Print some image information
	std::cout << "Image " << image_filepath << " succesfully loaded." << std::endl;
	std::cout << "Dimension: " << source_image.cols << "X" << source_image.rows<<std::endl;
	std::cout << "Number of Channels: " << source_image.channels() << std::endl;
	std::cout << "Step: "<<source_image.step<<std::endl;

	//========================TO DO:EXERCISE 1=========================
	//show the image in a window named "source image"

	//================================================================

	//stop the execution and wait for any key pressed by the user to resume
	std::cout << "Press any key while focusing on the GUI to continue..." << std::endl;
	cv::waitKey();

	//=======================TO DO: Exercise 2=========================
	//create an output iplimage with the same size of source_image and save the pointer in inverted_image
	//hint: cv::Mat as a function size() to get its size
	//hint: cv::Mat as a funciton type() to get its type

	//=================================================================

	if (inverted_image.data == NULL){
		std::cout << "Something went wrong during the creation of inverted_image, please check your code!" << std::endl;
		return 0;
	}
	else {
		//Print some image information
		std::cout << "Inverted image succesfully created. " << std::endl;
		std::cout << "Dimension: " << inverted_image.cols << "X" << inverted_image.rows << std::endl;
		std::cout << "Number of Channels: " << inverted_image.channels() << std::endl;
		std::cout << "Step: " << inverted_image.step << std::endl;
	}


	//=======================TO DO: Exercise 2=========================
	//Assign to each pixel of inverted_image 255-the corresponding pixel in source_image either using nested loops or linear algebra function.

	//=================================================================

	//========================TO DO:EXERCISE 2=========================
	//show the inverted image in a window named "inverted image"

	//=================================================================

	//stop the execution and wait for any key pressed by the user to resume, listen only to key pressed with focus on the GUI
	std::cout << "Press any key while focusing on the GUI to continue..." << std::endl;
	cv::waitKey();

	//========================TO DO:EXERCISE 3=========================
	//Modify all the code above to load a color image and invert all the three channels
	//=================================================================

	std::cout << "Execution Completed" <<std::endl;

	//destroy opencv GUI window
	cvDestroyAllWindows();

#endif
}