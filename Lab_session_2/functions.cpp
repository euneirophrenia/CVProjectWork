#include "functions.h"

#ifndef OPENCV_CPP_INTERFACE
//C Interface Function implementation
void resetHistogram(unsigned int * histogram){
	for (int i = 0; i<256; i++)
		histogram[i] = 0;
}

void visualizeHistogram(unsigned int * histogram){
	int outMargin = 20;
	int binw = 4;
	int binh = 3;
	int wout = (255 * binw) + (outMargin * 2);
	int hout = (100 * binh) + (outMargin * 2);
	CvSize size;
	size.width = wout;
	size.height = hout;
	IplImage * image_out = cvCreateImage(size, IPL_DEPTH_8U, 3);
	//draw output
	cvSet(image_out, CV_RGB(255, 255, 255));
	//draw axes
	CvPoint pt1, pt2;
	int thickness = 1;
	pt1.x = outMargin;
	pt1.y = outMargin;
	pt2.x = outMargin;
	pt2.y = hout - outMargin;
	cvLine(image_out, pt1, pt2, CV_RGB(0, 0, 0), thickness);
	pt1.x = wout - outMargin;
	pt1.y = hout - outMargin;
	cvLine(image_out, pt1, pt2, CV_RGB(0, 0, 0), thickness);

	//draw histo
	thickness = CV_FILLED;
	unsigned int max = 0;
	for (int i = 0; i<255; i++){
		if (histogram[i]>max)max = histogram[i];
	}
	for (int i = 0; i<255; i++){
		if (histogram[i]>0){
			pt1.x = outMargin + (i*binw);
			pt1.y = hout - outMargin;
			pt2.x = pt1.x + binw;
			pt2.y = pt1.y - ((histogram[i] * 100 / max)*binh);
			cvRectangle(image_out, pt1, pt2, CV_RGB(255 - i, i / 3, i), thickness);
		}
	}

	//font init
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_DUPLEX, 0.5f, 0.5f, 0, 2);
	pt1.x = outMargin;
	pt1.y = hout - (outMargin / 2);
	cvPutText(image_out, "0", pt1, &font, CV_RGB(0, 0, 0));
	pt1.x = wout - (1.5*outMargin);
	cvPutText(image_out, "255", pt1, &font, CV_RGB(0, 0, 0));

	//output visualization
	cvShowImage("Histogram", image_out);
	printf("Press any key to continue...\n");
	cvWaitKey();
	cvReleaseImage(&image_out);
	cvDestroyWindow("Histogram");
	return;
}


//TO DO for EXERCISE 1
void linearContrastStretch(IplImage * input, IplImage * output){
	//====================EXERCISE 1=========================
	//find max and min pixel value in input
	//set each pixel in output following the formulation on the slide
}

//TO DO for EXERCISE 2
void gammaCorrection(IplImage * input, IplImage * output, double r){
	//====================EXERCISE 2=========================
	//set each pixel in output following the formulation on the slide
}

//TO DO for EXERCISE 3
void imageEqualization(IplImage * input, IplImage * output, unsigned int*histogram){
	//====================EXERCISE 3==========================
	//implement histogram equalization following the formulation on the slide

}

//TO DO for EXERCISE 4
void convolution(IplImage * input, IplImage * output, float * kernel, int kernel_side){
	//=====================EXERCISE 4==========================
	//apply the convolutional operation on each pixel of input using the passed kernel, save the result in output
}
#else
//C++ interface function definition
void visualizeHistogram(const std::vector<unsigned int> & histogram){
	int outMargin = 30;
	int binw = 4;
	int binh = 3;
	int wout = (255 * binw) + (outMargin * 2);
	int hout = (100 * binh) + (outMargin * 2) ;
	cv::Size size;
	size.width = wout;
	size.height = hout;
	cv::Mat image_out = cv::Mat(size, CV_8UC3);
	image_out = cv::Scalar(255, 255, 255);

	//draw axes
	cv::line(image_out, cv::Point(outMargin, outMargin), cv::Point(outMargin, hout - outMargin), cv::Scalar(0, 0, 0), 2);
	cv::line(image_out, cv::Point(wout - outMargin, hout - outMargin), cv::Point(outMargin, hout - outMargin), cv::Scalar(0, 0, 0), 2);

	//draw histogram bin
	unsigned int max = 0;
	for (int i = 0; i<255; i++){
		if (histogram[i]>max)max = histogram[i];
	}
	for (int i = 0; i < 256; i++) {
		if (histogram[i] > 0){
			cv::Point pt1(outMargin + (i*binw), hout - outMargin);
			cv::Point pt2(pt1.x + binw, pt1.y - ((histogram[i] * 100 / max)*binh));
			cv::rectangle(image_out, pt1, pt2, cv::Scalar(255 - i, i / 3, i), CV_FILLED);
		}
	}

	//add notes
	std::string left = "0";
	std::string right = "255";
	cv::putText(image_out, left, cv::Point(outMargin, hout - (outMargin / 2)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
	cv::putText(image_out, right, cv::Point(wout - (1.5*outMargin), hout - (outMargin / 2)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);

	cv::imshow("histogram", image_out);
	std::cout << "Press any key to continue..." << std::endl;
	cv::waitKey();
	cv::destroyWindow("histogram");
}

//TO DO for EXERCISE 0
void buildHistogram(const cv::Mat & input, std::vector<unsigned int> & histogram){
	//====================EXERCISE 0========================
	//read the value of each pixel in the input image and increment by 1 the corresponding histogram cell
	//e.g. pixelvalue = 126 ---> histogram[126] = histogram[126]+1
	for (int row=0; row < input.rows; row++){
		for (int col=0; col < input.cols; col++) {
			histogram[input.data[row * (input.step) + col]]++;
		}
	}
}

//TO DO for EXERCISE 1
void linearContrastStretch(const cv::Mat & input, cv::Mat & output){
	//====================EXERCISE 1=========================
	//find max and min pixel value in input
	//set each pixel in output following the formulation on the slide
	int min, max;
	min = input.data[0];
	max = input.data[0];
	for (int row=0; row < input.rows; row++) {
		for (int col=0; col < input.cols; col++) {
			int datum = input.data[row * input.step + col];
			if ( datum < min)
				min = datum;
			if (datum > max)
				max = datum;
		}
	}

	for (int row=0; row < input.rows; row++) {
		for (int col=0; col < input.cols; col++) {
			output.data[row * output.step + col] = unsigned char ((255.0/(max-min)) * (input.data[row*input.step + col] - min));
		}
	}
}

//TO DO for EXERCISE 2
void gammaCorrection(const cv::Mat & input, cv::Mat & output, const double r){
	//====================EXERCISE 2=========================
	//set each pixel in output following the formulation on the slide
	for (int row=0; row < input.rows; row++) {
		for (int col=0; col < input.cols; col++) {
			output.data[row * output.step + col] =  (unsigned char)(std::pow(255, 1-r) * std::pow(input.data[row*input.step + col], r));
		}
	}
}

//TO DO for EXERCISE 3
void imageEqualization(const cv::Mat & input, cv::Mat & output, std::vector<unsigned int> & histogram){
	//====================EXERCISE 3==========================
	//implement histogram equalization following the formulation on the slide
	int factor = input.cols*input.rows;
	for (int row=0; row < input.rows; row++) {
		for (int col=0; col < input.cols; col++) {
			int tot=0;
			for (int k=0; k < input.data[row*input.step + col]; k++){
				tot+=histogram[k];
			}
			output.data[row * output.step + col] =  (unsigned char)(255.0/factor * (tot));
		}
	}

}

//TO DO for EXERCISE 4
void convolution(const cv::Mat & input, cv::Mat & output, std::vector<float> & kernel, int kernel_side){
	//=====================EXERCISE 4==========================
	//apply the convolutional operation on each pixel of input using the passed kernel, save the result in output
	int k=(kernel_side -1)/2;

	for (int row=0; row < input.rows; row++) {
		for (int col=0; col < input.cols; col++) {
			int result=0;
			for (int m=-k; m < k+1 ; m++) {
				for (int n=-k; n< k+1; n++){
					result += int(kernel[(m + k)*kernel_side + n + k]*(input.data[ (row - m)*input.step + (col - n)]));
				}
			}
			output.data[row * output.step + col] = unsigned char( result >=0 && result <=255 ? result : result < 0 ? 0 : 255);
		}
	}
}

#endif

