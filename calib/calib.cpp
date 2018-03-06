
#include <stdio.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <vector>
#include <sstream>
#include <iomanip>

// Creates data structure of 2D - 3D correspondences for calibration
void initPatterns3DPoints(int nFrames,  CvSize patternSize, float patternSquareSide, CvMat* & points, CvMat* & counts);

// Prints calibration results
void printCalib(CvMat* A, CvMat* d, CvMat* Rs = NULL, CvMat* Ts = NULL);

struct Points 
{
	Points() : p1(cvPoint(-1,-1)), p2(cvPoint(-1,-1)), enabled(false) {};

	CvPoint p1, p2;

	bool enabled;
};

void onMouseClick(int event, int x, int y, int flags, void* p)
{
	Points* points = reinterpret_cast<Points*>(p);

	if (points->enabled && event & CV_EVENT_LBUTTONUP)
	{
		if (points->p1.x == -1)
		{
			points->p1.x = x;
			points->p1.y = y;
		}
		else if (points->p2.x == -1)
		{
			points->p2.x = x;
			points->p2.y = y;
		}
		else
		{
			points->p1.x = points->p2.x ;
			points->p1.y = points->p2.y ;

			points->p2.x = x;
			points->p2.y = y;
		}
	}
}

int main(int argc, char** argv){

	/////////////////////////////////////////
	// MODIFY THESE PARAMETERS ACCORDING TO YOUR CHESSBOARD
	//////////////////////////////////////////
	const int internalCornerX = 5;
	const int internalCornerY = 8;
	const float patternSquareSide = 2.8f;
	
	const int imageWidth = 640;
	const int imageHeight = 480;
	

	

	CvSize patternSize = cvSize(internalCornerX, internalCornerY);
	

	const int nCorners =  internalCornerX * internalCornerY;


	// Intrisic parameters
	CvMat* A = NULL;
	CvMat* d = NULL;

	// Extrinsic parameters
	CvMat* Rs = NULL;
	CvMat* Ts = NULL;
	
	CvCapture* capture = cvCreateCameraCapture(0);
	bool camera = true;

	if (capture == NULL)
	{
		capture = cvCreateFileCapture("..\\videos\\calibVideo.avi");
		camera = false;
	}

	if (capture == NULL)
	{
		printf("Impossibile aprire camera o stream");
	}

	if (camera)
		cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, imageWidth);


	cvNamedWindow("Stream", 0);
	
	bool do_calibration = false, print_thirdXCorner = false, compute3DDistance = false;
	bool calibrated = false;
	bool moveToNextFrameInStreamMode = true;
	Points points2D;

	cvSetMouseCallback("Stream", onMouseClick, &points2D);

	int nCalibFrames = 0;
	std::vector<CvPoint2D32f> totCorner;
	IplImage* input;
	IplImage* output = NULL;
		
	
	while(true)
	{
		if (camera || moveToNextFrameInStreamMode)
		{
			input = cvQueryFrame(capture);
			if (output) cvReleaseImage(&output);
			if (input) output = cvCloneImage(input);
			moveToNextFrameInStreamMode = false;
		}
		
		if (output == NULL)
			break;
			

		char key = cvWaitKey(50);

		if (key == 'c')
		{
			printf("\nCalibration mode: press t to acquire a frame, d when done\n");
			do_calibration = true;
		}
		
		if (key == 'p')
		{
			printf("\n3D to 2D projection mode: press p again to exit\n");
			print_thirdXCorner = !print_thirdXCorner;
			compute3DDistance  = false;
		}

		if (key == 'b')
		{
			printf("\n2D to 3D back-projection mode: press b again to exit\n");
			points2D.enabled = compute3DDistance = !compute3DDistance;
			print_thirdXCorner = false;
		}

		if (key == 'n')
			moveToNextFrameInStreamMode = true;


		if (do_calibration && key == 'd')
		{
			CvMat* points;
			CvMat* counts;

			CvMat totCorner1Mat			= cvMat(1, nCalibFrames * nCorners, CV_32FC2, &totCorner[0]);
			
			initPatterns3DPoints(nCalibFrames, patternSize, patternSquareSide, points, counts);

			A = cvCreateMat(3, 3, CV_64FC1);
			d = cvCreateMat(5, 1, CV_64FC1);

			/////////////////////////////////////////
			// YOUR CODE HERE: see Exercise 1
			// create matrices to store rotation and translation vectors
			// when your done, modify the code at the end of the file to print them, too
			//////////////////////////////////////////
			Rs = NULL;
			Ts = NULL;
			//////////////////////////////////////////


			cvCalibrateCamera2(points, &totCorner1Mat, counts, cvGetSize(input), A, d, Rs, Ts);
			

			cvSave("A.yaml", A, "A");
			cvSave("d.yaml", d, "d");
			
			if (Rs != NULL && Ts != NULL)
			{
				cvSave("Rs.yaml", Rs, "Rs");
				cvSave("Ts.yaml", Ts, "Ts");
			}
			
			printf("\nCalibration saved\n");
			
			printCalib(A, d, Rs, Ts);

			do_calibration = false;
			calibrated = true;
			moveToNextFrameInStreamMode = true;
		}

		if (do_calibration)
		{
			CvPoint2D32f corners[nCorners];  
			int nFoundCorners = nCorners;

			int found = cvFindChessboardCorners(input, patternSize, corners, &nFoundCorners);
			
			if ((found || !camera) && key == 't')
			{
				IplImage* grayImg = cvCreateImage( cvGetSize(input), IPL_DEPTH_8U, 1 );

				if( input->nChannels == 1 )
					cvCopy( input, grayImg );
				else
					cvCvtColor( input, grayImg, CV_BGR2GRAY );

				cvFindCornerSubPix( grayImg, corners, nFoundCorners,
					cvSize(5,5), cvSize(-1,-1),
					cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,30,0.01));

				cvReleaseImage(&grayImg);

				char filename[1000];
				sprintf(filename, "calibFrame%02d.png", nCalibFrames);
				cvSaveImage(filename, input);

				nCalibFrames++;

				printf("Calibration Frames: %d\r", nCalibFrames);

				totCorner.insert(totCorner.end(), corners, corners + nFoundCorners);

				moveToNextFrameInStreamMode = true;
			}

			cvDrawChessboardCorners(output, patternSize, corners, nFoundCorners, found);
		}

		

		if (key == 'l')
		{
			A = (CvMat*)cvLoad("A.yaml");		
			d = (CvMat*)cvLoad("d.yaml");

			Rs = (CvMat*)cvLoad("Rs.yaml");		
			Ts = (CvMat*)cvLoad("Ts.yaml");
		
			printCalib(A,d,Rs,Ts);
			calibrated = true;
		}


		if (!do_calibration && !print_thirdXCorner && !compute3DDistance)
		{
			if (camera)
				printf("(c) calibration, (l) load, (q) exit, (p) ex2, (b) ex3\r");
			else
				printf("(c) calibration, (l) load, (n) next frame, (q) exit, (p) ex2, (b) ex3\r");
		}

		if (calibrated && !do_calibration)
		{
			if (!print_thirdXCorner && !compute3DDistance)
			{
				IplImage* img1Undist = cvCloneImage(output);
				cvUndistort2( output, img1Undist, A, d);
				
				cvShowImage("Stream", img1Undist);
				cvReleaseImage(&img1Undist);

			}
			else
			{
				int imageIdx = 0;
				std::stringstream sstr;
				sstr << "calibFrame" << std::setw(2) << std::setfill('0') << imageIdx << ".png";
				IplImage* calibFrame = cvLoadImage(sstr.str().c_str());
				IplImage* calibFrameUndist = NULL;
				
				if (calibFrame != NULL)
				{
					calibFrameUndist = cvCloneImage(calibFrame);
					cvUndistort2( calibFrame, calibFrameUndist, A, d);

					if (print_thirdXCorner)
					{
						CvPoint point2D = cvPoint(calibFrameUndist->width/2,calibFrameUndist->height/2);

						/////////////////////////////////////////
						// YOUR CODE HERE: see exercise 2, 
						// you should compute the 2D coordinates of the 3D point (3 * squareSide, 1 * squareSide, 0)
						// and set the x and y fields of point2D them on the IplImage contained in the variable firstFrameUndist
						/////////////////////////////////////////
						

						////////////////////////////////////////////
						cvCircle(calibFrameUndist, point2D, 4, CV_RGB(255,0,0), CV_FILLED);
					}
					else //compute3DDistance == true
					{
						double distance = 0.0;

						if (points2D.p1.x != -1 && points2D.p2.x != -1)
						{
						
							/////////////////////////////////////////
							// YOUR CODE HERE : see Exercise 3,
							// you should compute the 3D coordinates of the two CvPoints 
							// points2D.p1 and points2D.p2 from calibration frame, 
							// then compute the 3D distance between the backprojections 
							// and put the result in distance
							/////////////////////////////////////////

				
							/////////////////////////////////////////
						}

						if (points2D.p1.x != -1)
						{
							cvCircle(calibFrameUndist, points2D.p1, 2, CV_RGB(0, 255, 0), CV_FILLED);
						}

						if (points2D.p2.x != -1)
						{
							cvCircle(calibFrameUndist, points2D.p2, 2, CV_RGB(0, 255, 0), CV_FILLED);
						}

						if (points2D.p1.x != -1 && points2D.p2.x != -1)
						{
							cvLine(calibFrameUndist, points2D.p1, points2D.p2, CV_RGB(0, 255, 0), 2);
							CvFont f = cvFont(2,2);
							std::stringstream d_str;
							d_str << distance << " cm";
							cvPutText(calibFrameUndist, d_str.str().c_str(), points2D.p1, &f, CV_RGB(0,255,0));
						}
						
					}
					
				}
				
				if (calibFrame != NULL)
				{
					cvNamedWindow("Stream", 0);
					cvShowImage("Stream",calibFrameUndist);

					cvReleaseImage(&calibFrame);
					cvReleaseImage(&calibFrameUndist);
				}
				else 
				{
					print_thirdXCorner = false;
					compute3DDistance = false;
				}

			}
			
		}
		else
		{
			cvShowImage("Stream", output);
		}

		if (key == 'q')
			break;
	}

	printf("\n");
	cvReleaseCapture(&capture);
}

void initPatterns3DPoints(int nFrames,  CvSize patternSize, float patternSquareSide, CvMat* & points, CvMat* & counts)
{
	std::vector<CvPoint2D32f> patternPoints;

	for( int y = 0; y < patternSize.height; y++ )
		for( int x = 0; x < patternSize.width; x++ )
		{
			patternPoints.push_back(cvPoint2D32f( x*patternSquareSide,
				y*patternSquareSide ));
		}

	int patternPointCount = (int)patternPoints.size();

	points = cvCreateMat(1, nFrames * patternPointCount, CV_32FC3);

	counts = cvCreateMat( 1, nFrames, CV_32S);

	for( int k = 0; k < nFrames; k++ )
	{
		counts->data.i[k] = patternPointCount;

		for( int j = 0; j < patternPointCount; j++ )
		{

			int row = (k * patternPointCount + j)*3;
			points->data.fl[row + 0] = patternPoints[j].x;
			points->data.fl[row + 1] = patternPoints[j].y;
			points->data.fl[row + 2] = 0.0;
		}
	}
}

void printCalib(CvMat* A, CvMat* d, CvMat* Rs, CvMat* Ts)
{
	printf("\nCalibration loaded\n");
	printf("    | %.3lf %.3lf %.3lf |\n", A->data.db[0], A->data.db[1], A->data.db[2]);
	printf("A = | %.3lf %.3lf %.3lf |\n", A->data.db[3], A->data.db[4], A->data.db[5]);
	printf("    | %.3lf %.3lf %.3lf |\n", A->data.db[6], A->data.db[7], A->data.db[8]);

	printf("D (radial) : %.3lf %.3lf %.3lf \n", d->data.db[0], d->data.db[1], d->data.db[4]);
	printf("D (tangent) : %.5lf %.5lf \n", d->data.db[2], d->data.db[3]);

	if (Rs != NULL && Ts != NULL)
	{
		/////////////////////////////////////////
		// YOUR CODE HERE: see Exercise 1,
		// print extrinsic parameters 
		// REMEMBER: a pair (R,T) for each calibration image, 
		// there are as many calibration images as Rs or Ts rows
		//////////////////////////////////////////

		
		/////////////////////////////////////////

	}
}


