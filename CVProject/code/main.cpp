//
// Created by Marco DiVi on 26/04/18.
//

#include "processing/preprocessing.h"
#include "processing/GHTMatching.h"

//#define USE_GHT
#define TEST_SCENE "scenes/e5.png"


int main(int argc, char** argv){

/*--------------------- OFFLINE preprocessing ---------------*/
    Preprocesser preprocesser; //will handle the models for us

    ///equalize size of the models and blur where needed
    preprocesser.uniform(-1); //-1 => automatic size decision, any value >0 will use that value instead

    /// setup the algorithm
    auto detector = cv::xfeatures2d::SIFT::create(); //(0, 3, 0.04, 15);

    /// For SIFT / SURF
    auto indexparams = new cv::flann::KDTreeIndexParams(context.KDTREES_INDEX);

    cv::flann::SearchParams* searchParams = new cv::flann::SearchParams((int)context["FLANN_SEARCH_ITERATIONS"]);
    cv::FlannBasedMatcher* matcher = new cv::FlannBasedMatcher(indexparams, searchParams);
    Algorithm* alg = new Algorithm(detector, matcher);

    ///compute features on the loaded models
    preprocesser.build(alg);  //extracts features AND builds the hough model

    ///compute their similarity and label the models accordingly (they will have the isHard flag set)
    preprocesser.computeSimilarity(alg, 0.15);  //use 0.1 as threshold for confusing models, can be set higher

    auto hardModels = preprocesser.hardModels(); //retrieve hard and easy models as computed before
    auto easyModels = preprocesser.easyModels();

/*---------------- Start of the runtime processing ------------------- */
#ifdef TIMEIT
    ///start timing, starting the system clock
    auto now = std::chrono::high_resolution_clock::now();
#endif

    ///load the scene both in gray scale and in color, since it will be used to display the output
    auto scene = new RichImage(context.BASE_PATH + TEST_SCENE);
    auto colorscene = cv::imread(context.BASE_PATH + TEST_SCENE, cv::IMREAD_COLOR);

    //scene->deBlur(true); //deblur being "fast" or not, not useful
    scene->build(alg, false); //false, because we don't need the hough model for the scene

    int approx_scale = scene->approximateScale(); //computed only once, even if we didn't save it, it is cached within

	Logger::log("Detected scale:\t" + std::to_string(approx_scale));

    auto ghtmatcher = new GHTMatcher(scene); //initialize the matcher, by default it uses the approxScale of the scene

#ifdef TIMEIT
	auto c_start = std::clock();
	auto begin_match = std::chrono::high_resolution_clock::now();
#endif

/*------------ Perform simple SIFT match --------------*/
#ifndef USE_GHT
	//auto multi = multiMatch(preprocesser.models(), scene, *alg, false); // slow only
	auto multi = multiMatch(preprocesser.easyModels(), scene, *alg, true); //easy version
	auto tmp = multiMatch(preprocesser.hardModels(), scene, *alg, false); // hard version
	multi.insert(tmp.begin(), tmp.end()); //merge results of the two steps
#endif

/*----------- Perform GHT based matching ------------ */
#ifdef USE_GHT
    ghtmatcher->GHTMatch(hardModels, alg); //use the slow but robust variation on the hard models
    auto multi = ghtmatcher->FastGHTMatch(easyModels, alg);  //use the fast variation on the "easy" models
#endif

#ifdef TIMEIT
	auto c_end = std::clock();
	std::cerr << "[DEBUG] Match CPU time: " << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << " ms\n";
	std::chrono::duration<double> end_match = std::chrono::high_resolution_clock::now() - begin_match;
	std::cerr << "[DEBUG] Match absolute time: " << end_match.count() << " seconds\n";
	std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - now;
	std::cerr << "[DEBUG] Execution completed in " << elapsed.count() << " seconds\n";
#endif

/* ---------------- GHT Result display --------------------- */
#ifdef USE_GHT
	for (auto match : multi) {
		auto ghtmatch = match.second;
		std::string modelname = match.first -> path;
		std::cout << "\nLooking for " << modelname << "(isHard: " << match.first->isHard <<")...\n";

		if (!ghtmatch.empty()) {
			for (auto blob : ghtmatch) {
				std::cout << "\tFound at " << blob.position << "\t(confidence: " << blob.confidence << ")\n";
				//cv::drawMarker(colorscene, blob.position, colors[blob.modelName]);


				cv::putText(colorscene, preprocesser.idOf(blob.model->path), blob.position,
							cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, CvScalar(250, 255,250));

				auto rect = boundingRect(blob.model, scene, blob.matches);
				if (!rect.empty())
					cv::rectangle(colorscene, rect, CvScalar(0, 255, 0), 2, cv::LINE_AA);
				else
					std::cout << "\t\tCould not find proper homography for: " << blob << "\n";

			}
		}
	}
#endif
#ifndef USE_GHT
/*------ Normal SIFT results display ------*/
	for (auto pair : multi) {
		std::string modelname = pair.first -> path;
		std::cout << "\nLooking for " << modelname << "(total matches: " << pair.second.size() <<")...\n";
		if (pair.second.size() > 110) {
			auto rect = boundingRect(pair.first, scene, pair.second);
			if (!rect.empty()) {
				cv::Point center = (rect.br() + rect.tl()) * 0.5;
				cv::putText(colorscene, preprocesser.idOf(modelname), center,
							cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, CvScalar(250, 255, 250));
				cv::rectangle(colorscene, rect, CvScalar(0, 255, 0), 2, cv::LINE_AA);
				std::cout << "\tFound at " << center << "\n";
			} else
				std::cout << "\t\tCould not find proper homography for: " << modelname << "\n";
		}
	}
#endif

	cv::imshow("Matches", colorscene);
	cv::waitKey(0);


    return 0;

}