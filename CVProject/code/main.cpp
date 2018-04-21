//
// Created by Marco DiVi on 27/02/18.
//

#include "processing/GHTMatching.h"
#ifdef DEBUG
    #include <ctime>
    #include <chrono>
#endif

//#include "processing/InfiniteMatrix.h"

#define TEST_SCENE "scenes/m5.png"


int main(int argc, char** argv){

    std::map<std::string, std::string> ids;

    auto msize = context.MODELS.size();

    for (int i =0; i< msize; i++) {
        ids[context.MODELS[i]] = fileName(context.MODELS[i], true);
    }

    /// setup the algorithm
    auto detector = cv::xfeatures2d::SIFT::create(); //(0, 3, 0.04, 15);
    //auto detector = cv::xfeatures2d::SURF::create(context["SURF_MIN_HESSIAN"]);
    //auto detector = cv::BRISK::create((int)context["BRISK_THRESHOLD"], (int)context["BRISK_OCTAVES"], (float)context["BRISK_PATTERNSCALE"]);
    //auto detector = cv::ORB::create();


    /// For SIFT / SURF
    auto indexparams = new cv::flann::KDTreeIndexParams(context.KDTREES_INDEX);

    /// FOR BRISK / ORB
    /*auto indexparams = new cv::flann::LshIndexParams((int)context["LSH_INDEX_TABLES"],
                          (int) context["LSH_INDEX_KEY_SIZE"], (int) context["LSH_INDEX_MULTIPROBE_LEVEL"]);*/


    cv::flann::SearchParams* searchParams = new cv::flann::SearchParams((int)context["FLANN_SEARCH_ITERATIONS"]);
    cv::FlannBasedMatcher* matcher = new cv::FlannBasedMatcher(indexparams, searchParams);


    Algorithm* alg = new Algorithm(detector, matcher);

    std::vector<RichImage*> model_references;
    for (auto& p : context.MODELS) {
        model_references.push_back(new RichImage(p, cv::IMREAD_GRAYSCALE));
    }

    auto scene = new RichImage(context.BASE_PATH + TEST_SCENE); //Images.getOrElse(context.BASE_PATH + TEST_SCENE, load(cv::IMREAD_GRAYSCALE));
    auto colorscene = cv::imread(context.BASE_PATH + TEST_SCENE, cv::IMREAD_COLOR);

    ///uniform models (with automatic setting could be done "offline")
    uniform(model_references, -1); // automatic size decision
    //uniform(model_references, approx_scale);  // given size

    ///extract the features and also extract the ght model
    for (auto m : model_references){
        m->build(alg, true);
    }


#ifdef DEBUG
    auto now = std::chrono::high_resolution_clock::now();
    std::clock_t c_start = std::clock();
#endif
    ///preprocess the scene
    scene->deBlur(false);
    scene -> build(alg);
    int approx_scale = scene->approximateScale();

#ifdef DEBUG
    std::clock_t c_end = std::clock();
    std::cerr << "[DEBUG] Scene:\t" << scene->path << "\n";
    std::cerr << "[DEBUG] Detected scale:\t" << approx_scale << "\n";
    std::cerr << "[DEBUG] Scene preprocessing CPU time: " << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << " ms\n";
#endif

    auto ghtmatcher = new GHTMatcher(scene);

#ifdef FAST
#ifdef DEBUG
    c_start = std::clock();
#endif
    auto multi = ghtmatcher->FastGHTMatch(model_references, alg);
#ifdef DEBUG
    c_end = std::clock();
    std::cerr << "[DEBUG] FastGHTMatch CPU time: " << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << " ms\n";
#endif
#else
    #ifdef DEBUG
        c_start = std::clock();
    #endif
    auto multi = ghtmatcher->GHTMatch(model_references, alg);
    #ifdef  DEBUG
        c_end = std::clock();
        std::cerr << "[DEBUG] GHTMatch CPU time: " << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << " ms\n";
    #endif
#endif

#ifdef DEBUG

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - now;
    std::cerr << "[DEBUG] Execution completed in " << elapsed.count() << " seconds\n";

#endif

    for (auto match : multi) {
        auto ghtmatch = match.second;
        std::string modelname = match.first -> path;
        std::cout << "\nLooking for " << modelname << "...\n";

        if (!ghtmatch.empty()) {
            for (auto blob : ghtmatch) {
                std::cout << "\tFound at " << blob.position << "\t(confidence: " << blob.confidence << ")\n";
                //cv::drawMarker(colorscene, blob.position, colors[blob.modelName]);


                cv::putText(colorscene, ids[blob.model->path], blob.position,
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, CvScalar(250, 255,250));

                auto rect = boundingRect(blob.model, scene, blob.matches);
                if (!rect.empty())
                    cv::rectangle(colorscene, rect, CvScalar(0, 255, 0), 2, cv::LINE_AA);
                else
                    std::cout << "\t\tCould not find proper homography for: " << blob << "\n";

            }
        }
    }
    cv::imshow("Matches", colorscene);
    cv::waitKey(0);

}