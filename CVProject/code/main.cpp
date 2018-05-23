//
// Created by Marco DiVi on 26/04/18.
//

#include "processing/preprocessing.h"
#include "processing/GHTMatching.h"
#include "utilities/cli.h"

void displayGHTResults(std::map<RichImage*, std::vector<Blob>> results, cv::Mat* colorscene, Preprocesser preprocesser, RichImage* scene) {
	for (auto match : results) {
		auto ghtmatch = match.second;
		std::string modelname = match.first->path;
		std::cout << "\nLooking for " << modelname << "\t(isHard: " << match.first->isHard << ")...\n";

		if (!ghtmatch.empty()) {
			for (auto blob : ghtmatch) {

				auto rect = boundingRect(blob.model, scene, blob.matches);
				if (!rect.empty()) {
					cv::Point center = (rect.br() + rect.tl()) * 0.5;
					cv::rectangle(*colorscene, rect, CvScalar(0, 255, 0), 2, cv::LINE_AA);
					std::cout << "\tFound at " << center << "\t(confidence: " << blob.confidence << ", width: " << rect.width << ", height: "<< rect.height <<")\n";
					//cv::drawMarker(colorscene, blob.position, colors[blob.modelName]);


					cv::putText(*colorscene, preprocesser.idOf(blob.model->path), center,
								cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, CvScalar(250, 255, 250));
				}
				else
					std::cout << "\tCould not find proper homography for: " << blob << "\n";

			}
		}
	}
}

void displayResults(std::map<RichImage*, std::vector<cv::DMatch>> results, cv::Mat* colorscene, Preprocesser preprocesser, RichImage* scene) {
	for (auto pair : results) {
		std::string modelname = pair.first->path;
		std::cout << "\nLooking for " << modelname << "\t(isHard: " << pair.first->isHard << "\ttotal matches: " << pair.second.size() << ")...\n";
		if (pair.second.size() > 110) {
			auto rect = boundingRect(pair.first, scene, pair.second);
			if (!rect.empty()) {
				cv::Point center = (rect.br() + rect.tl()) * 0.5;
				cv::putText(*colorscene, preprocesser.idOf(modelname), center,
							cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, CvScalar(250, 255, 250));
				cv::rectangle(*colorscene, rect, CvScalar(0, 255, 0), 2, cv::LINE_AA);
				std::cout << "\tFound at " << center << "\t(width: " << rect.width << ", height: "<< rect.height <<")\n";
			} else
				std::cout << "\t\tCould not find proper homography for: " << modelname << "\n";
		}
	}
}

int main(int argc, char** argv) {

	/* Some variable setup */
	std::string test_scene_path;
	detect(argc, argv, &test_scene_path);

	Logger::logLevel = (context.cli_options->at("-debug")? DEBUG : INFO);
	if (context.cli_options->at("-debug"))
		Logger::log("DEBUG MODE ON: results displayed may be incorrect.\n"
			  "\tIn order to ease debug, some 'tricks' on memory were performed"
	 " which may lead to inconsistent displaying of results (as can be seen disabling debug mode).", "[WARNING]\t", WARNING);

	std::map<RichImage*, std::vector<Blob>> ghtResults;
	std::map<RichImage*, std::vector<cv::DMatch>> results;

	clock_t c_start;
	auto begin_match = std::chrono::high_resolution_clock::now(), now = std::chrono::high_resolution_clock::now();


/*--------------------- OFFLINE preprocessing ---------------*/
	Preprocesser preprocesser; //will handle the models for us

	///equalize size of the models and blur where needed
	preprocesser.uniform(-1); //-1 => automatic size decision, any value >0 will use that value instead

	/// setup the algorithm
	auto detector = cv::xfeatures2d::SIFT::create(); //(0, 3, 0.04, 15) default values tuned by Lowe

	/// For SIFT / SURF
	auto indexparams = new cv::flann::KDTreeIndexParams(context.KDTREES_INDEX);

	cv::flann::SearchParams* searchParams = new cv::flann::SearchParams((int) context["FLANN_SEARCH_ITERATIONS"]);
	cv::FlannBasedMatcher* matcher = new cv::FlannBasedMatcher(indexparams, searchParams);
	Algorithm* alg = new Algorithm(detector, matcher);

	///compute features on the loaded models
	preprocesser.build(alg);  //extracts features AND builds the hough model

	///compute their similarity and label the models accordingly (they will have the isHard flag set)
	preprocesser.computeSimilarity(alg, 0.15);  //use 0.1 as threshold for confusing models, can be set higher

	auto hardModels = preprocesser.hardModels(); //retrieve hard and easy models as computed before
	auto easyModels = preprocesser.easyModels();

/*---------------- Start of the runtime processing ------------------- */
	if (context.cli_options->at("-time")) {
		///start timing, starting the system clock
		now = std::chrono::high_resolution_clock::now();
	}

	///load the scene both in gray scale and in color, since it will be used to display the output
	auto scene = new RichImage(context.BASE_PATH + test_scene_path);
	auto colorscene = cv::imread(context.BASE_PATH + test_scene_path, cv::IMREAD_COLOR);

//	scene->deBlur(false); //deblur being "fast" or not, not useful
	scene->build(alg, false); //false, because we don't need the hough model for the scene, only extract features

	int approx_scale = scene->actualScale(); //computed only once, even if we didn't save it, it is cached within

	Logger::log("Detected scale:\t" + std::to_string(approx_scale), "[INFO]\t", INFO);

	auto ghtmatcher = new GHTMatcher(scene); //initialize the matcher, by default it uses the approxScale of the scene

	if (context.cli_options->at("-time")) {
		c_start = std::clock();
		begin_match = std::chrono::high_resolution_clock::now();
	}

/*------------ Perform simple SIFT match --------------*/
	if (!context.cli_options->at("-ght")){
		//auto multi = multiMatch(preprocesser.models(), scene, *alg, false); // slow only
		results = multiMatch(preprocesser.easyModels(), scene, *alg, true); //easy version
		auto tmp = multiMatch(preprocesser.hardModels(), scene, *alg, false); // hard version
		results.insert(tmp.begin(), tmp.end()); //merge results of the two steps
}

/*----------- Perform GHT based matching ------------ */
	else {
		ghtmatcher->GHTMatch(hardModels, alg); //use the slow but robust variation on the hard models
		ghtResults = ghtmatcher->FastGHTMatch(easyModels, alg);  //use the fast variation on the "easy" models
	}

	if (context.cli_options->at("-time")) {
		auto c_end = std::clock();
		std::cerr << "[DEBUG] Match CPU time: " << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms\n";
		std::chrono::duration<double> end_match = std::chrono::high_resolution_clock::now() - begin_match;
		std::cerr << "[DEBUG] Match absolute time: " << end_match.count() << " seconds\n";
		std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - now;
		std::cerr << "[DEBUG] Execution completed in " << elapsed.count() << " seconds\n";
	}

/* ---------------- GHT Result display --------------------- */
	if (context.cli_options->at("-ght")) {
		displayGHTResults(ghtResults, &colorscene, preprocesser, scene);
	}
	else {
/*------ Normal SIFT results display ------*/
		displayResults(results, &colorscene, preprocesser, scene);
	}

	cv::imshow("Matches", colorscene);
	cv::waitKey(0);


    return 0;

}