#include "functions.h"

int main(int argc, char**argv){
#ifndef OPENCV_CPP_INTERFACE
	printf("*************************************************\n");
	printf("**                Histograms                   **\n");
	printf("*************************************************\n");

	//set in img_path the path to the image that you want to load
	char* imgPath= "../data/pum1dim1.bmp";
	IplImage * source_image = 0;

	printf("Tryiing to open: %s\n", imgPath);
	source_image = cvLoadImage(imgPath,CV_LOAD_IMAGE_GRAYSCALE);

	//Check if the image is properly laoded
	if (source_image == NULL){
		printf("Unable to load image with file path: %s\n", imgPath);
		printf( "Ending execution..." );
		return 0;
	}

	//Show the original image and print some information
	cvShowImage("original", source_image);
	printf("Image %s succesfully loaded\n", imgPath);
	printf("Dimension: %dX%d\n",source_image->width, source_image->height);
	printf("Number of channels: %d\n", source_image->nChannels);
	printf("Step: %d\n", source_image->widthStep);
	printf("==========================================\n");

	//====================EXERCISE 0=================================
	//Create a vector of 256 elements initialized at zero to handle the image histogram
	printf("Going to build image histogram...\n");
	unsigned int histogram[256];
	resetHistogram(histogram);
	buildHistogram(source_image, histogram);
	//===============================================================

	//show the histogram created
	visualizeHistogram(histogram);

	bool exit = false;
	while (!exit){
		int choice = 0;
		printf("\nWhat do you want to do: 1)linear contrast stretch 2)gamma correction 3)equalization 4)convolution 5)exit\n");
		printf("Enter you choice: ");
		scanf("%d", &choice);

		if (choice == 1){
			//==========================EXERCISE 1==========================
			//Apply linear contrast stretch
			printf( "Going to apply linear contrast stretch...\n");
			IplImage * linear_constrast = cvCreateImage(cvGetSize(source_image), source_image->depth, source_image->nChannels);
			linearContrastStretch(source_image, linear_constrast);
			//==============================================================

			//Show strectched image
			cvShowImage("Linear Contrast Stretch", linear_constrast);
			unsigned int linear_histogram[256];
			resetHistogram(linear_histogram);
			buildHistogram(linear_constrast, linear_histogram);
			visualizeHistogram(linear_histogram);
			cvDestroyWindow("Linear Contrast Stretch");
		}
		else if (choice == 2){
			//==========================EXERCISE 2==========================
			//Apply gamma correction
			printf("Going to apply gamma correction...\n");
			double r = 0.5;
			IplImage * gamma_corrected = cvCreateImage(cvGetSize(source_image), source_image->depth, source_image->nChannels);
			gammaCorrection(source_image, gamma_corrected, r);
			//==============================================================

			//Show gamma corrected image
			cvShowImage("Gamma corrected", gamma_corrected);
			unsigned int gamma_histogram[256];
			resetHistogram(gamma_histogram);
			buildHistogram(gamma_corrected, gamma_histogram);
			visualizeHistogram(gamma_histogram);
			cvDestroyWindow("Gamma corrected");
		}
		else if (choice == 3){
			//==========================EXERCISE 3==========================
			//Apply histogram equalization
			printf("Going to apply histogram equalization...\n");
			IplImage * equalized = cvCreateImage(cvGetSize(source_image), source_image->depth, source_image->nChannels);
			imageEqualization(source_image, equalized, histogram);
			//==============================================================

			//Show equalized image
			cvShowImage("Equalized", equalized);
			unsigned int equalized_histogram[256];
			resetHistogram(equalized_histogram);
			buildHistogram(equalized, equalized_histogram);
			visualizeHistogram(equalized_histogram);
			cvDestroyWindow("Equalized");
		}
		else if (choice == 4){
			//==========================EXERCISE 4==========================
			//Apply convolutional kernel
			printf("Going to apply histogram equalization...\n");
			IplImage * convolved = cvCreateImage(cvGetSize(source_image), source_image->depth, source_image->nChannels);
			int kernel_side = 3;
			//uncomment only one of the two kernel at time
			float kernel[9] = { 1/16.0, 2/16.0, 1/16.0, 2/16.0, 4/16.0, 2/16.0, 1/16.0, 2/16.0, 1/16.0 };
			//float kernel[9] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
			convolution(source_image, convolved, kernel,kernel_side);
			//==============================================================

			//Show convolved image
			cvShowImage("convolved", convolved);
			unsigned int convolved_histogram[256];
			resetHistogram(convolved_histogram);
			buildHistogram(convolved, convolved_histogram);
			visualizeHistogram(convolved_histogram);
			cvDestroyWindow("convolved");
		}
		else if (choice == 5){
			exit = true;
		} else {
			printf("Unrecognized Option...\n");
		}
	}
	printf("Bye Bye...");
#else
	std::cout << "*************************************************" << std::endl;
	std::cout << "**                Histograms                   **" << std::endl;
	std::cout << "*************************************************" << std::endl;

	//set in img_path the path to the image that you want to load
	std::string img_path = "../data/cln1.bmp";
	std::cout << "Trying to open: '" << img_path << "'..." << std::endl;

	//load source image
	cv::Mat source_image = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);

	//Check if the image is properly loaded
	if (source_image.data == NULL){
		std::cout << "Unable to load image with file path: " << img_path << std::endl;
		std::cout << "Ending execution..." << std::endl;
		return 0;
	}

	//Show the original image and print some information
	cv::imshow("original", source_image);
	std::cout << "Image " << img_path << " succesfully loaded." << std::endl;
	std::cout << "Dimension: " << source_image.cols << "X" << source_image.rows << std::endl;
	std::cout << "Number of Channels: " << source_image.channels() << std::endl;
	std::cout << "Step: " << source_image.step << std::endl;
	std::cout << "===========================================" << std::endl;

	//====================EXERCISE 0=================================
	//Create a vector of 256 elements initialized at zero to handle the image histogram
	std::cout << "Going to build image histogram..." << std::endl;
	std::vector<unsigned int> histogram(256, 0);
	buildHistogram(source_image, histogram);
	//===============================================================

	//show the histogram created
	visualizeHistogram(histogram);

	bool exit = false;
	while (!exit){
		int choice=0;
		std::cout << std::endl<<"What do you want to do: 1)linear contrast stretch 2)gamma correction 3)equalization 4)convolution 5)exit" << std::endl;
		std::cout << "Enter you choice: ";
		std::cin >> choice;

		if (choice == 1){
			//==========================EXERCISE 1==========================
			//Apply linear contrast stretch
			std::cout << "Going to apply linear contrast stretch..." << std::endl;
			cv::Mat linear_constrast = cv::Mat::zeros(source_image.size(), source_image.type());
			linearContrastStretch(source_image, linear_constrast);
			//==============================================================

			//Show strectched image
			cv::imshow("Linear Contrast Stretch", linear_constrast);
			std::vector<unsigned int> linear_histogram(256, 0);
			buildHistogram(linear_constrast, linear_histogram);
			visualizeHistogram(linear_histogram);
			cv::destroyWindow("Linear Contrast Stretch");
		}
		else if (choice == 2) {
			//=========================EXERCISE 2===========================
			//Apply gamma correction
			std::cout << "Going to apply gamma correction..." << std::endl;
			cv::Mat gamma_correction = cv::Mat::zeros(source_image.size(), source_image.type());
			double r = 1.5;
			gammaCorrection(source_image, gamma_correction, r);
			//==============================================================

			//Show gamma corrected image
			cv::imshow("Gamma corrected", gamma_correction);
			std::vector<unsigned int> gamma_histogram(256, 0);
			buildHistogram(gamma_correction, gamma_histogram);
			visualizeHistogram(gamma_histogram);
			cv::destroyWindow("Gamma corrected");
		}
		else if (choice == 3){
			//=========================EXERCISE 3===========================
			//Apply histogram equalization
			std::cout << "Going to apply histogram equalization..." << std::endl;
			cv::Mat equalized = cv::Mat::zeros(source_image.size(), source_image.type());
			imageEqualization(source_image, equalized,histogram);
			//==============================================================

			//Show equalized image
			cv::imshow("Equalized", equalized);
			std::vector<unsigned int> equalized_histogram(256, 0);
			buildHistogram(equalized, equalized_histogram);
			visualizeHistogram(equalized_histogram);
			cv::destroyWindow("Equalized");
		}
		else if (choice == 4){
			//=========================EXERCISE 4===========================
			//Apply convolution
			std::cout << "Going to apply histogram convolutional filters on the image..." << std::endl;
			int kernel_side = 3;
			std::vector<float> kernel(kernel_side*kernel_side);
			//uncomment only one of the two kernel at time
			//float k[9] = { 1/16.0, 2/16.0, 1/16.0, 2/16.0, 4/16.0, 2/16.0, 1/16.0, 2/16.0, 1/16.0 };
			float k[9] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
			kernel.assign(k,k+9);
			cv::Mat convolved = cv::Mat::zeros(source_image.size(), source_image.type());
			convolution(source_image, convolved, kernel,kernel_side);
			//==============================================================

			//Show convolved image
			cv::imshow("Convolved", convolved);
			std::vector<unsigned int> convolved_histogram(256, 0);
			buildHistogram(convolved, convolved_histogram);
			visualizeHistogram(convolved_histogram);
			cv::destroyWindow("Convolved");
		}
		else if (choice == 5){
			exit = true;
		}
		else {
			std::cout << "Unrecognized option" << std::endl;
		}
	}
	std::cout << "Bye Bye..." << std::endl;
#endif
}