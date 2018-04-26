//
// Created by Marco DiVi on 27/02/18.
//

#include "processing/GHTMatching.h"
#include "processing/template.h"
#ifdef DEBUG
    #include <ctime>
    #include <chrono>
#endif

//#include "processing/InfiniteMatrix.h"

#define TEST_SCENE "scenes/e3.png"

void measure() {
    float N = 15.0f;
    auto now = std::chrono::high_resolution_clock::now();
    for (int i=0; i< N; i++) {

    }
    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - now;
    std::cerr << "[MEASURE] avareage over " << N <<" cycles: " << elapsed.count()/N << " seconds\n";
}


int oldmain(int argc, char** argv){

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

    auto scene = new RichImage(context.BASE_PATH + TEST_SCENE);
    auto colorscene = cv::imread(context.BASE_PATH + TEST_SCENE, cv::IMREAD_COLOR);

#ifndef USE_TEMPLATE
    ///uniform models (with automatic setting could be done "offline")
    uniform(model_references, -1); // automatic size decision

    ///extract the features and also extract the ght model
    for (auto m : model_references){
        m->build(alg, true);
    }

#else
    ///to the approximate scale,
    uniform(model_references, scene->approximateScale());
#endif


#ifdef USE_SIMILARITY
    float** similarity;
    similarity = new float*[model_references.size()];
    std::vector<RichImage*> easyToTell, hardToTell;

    bool placed[model_references.size()];

    for (int i=0; i<model_references.size(); i++) {
        similarity[i] = new float[model_references.size()];
    }


    for (int i=0; i<model_references.size(); i++) {
        similarity[i][i] = 1.0f;
        if (model_references[i]->isHard)
            continue;

        for (int j=i+1; j < model_references.size(); j++) {
           if (model_references[j]->isHard)
               continue;

           auto matches = findKnn(model_references[i]->features, model_references[j]->features, matcher);
           similarity[i][j] = 1.0f* matches.size() / model_references[i]->features.rows;
           similarity[j][i] = 1.0f* matches.size() / model_references[j] -> features.rows;
#ifdef DEBUG
           if (similarity[i][j] > 0.1)
               std::cerr <<"[SIMILARITY] " << model_references[i]->path << "\t" << model_references[j]->path << "\t" << similarity[i][j] << "\n";
           if (similarity[j][i] > 0.1)
               std::cerr <<"[SIMILARITY] " << model_references[j]->path << "\t" << model_references[i]->path << "\t" << similarity[j][i] << "\n";
#endif
            if (similarity[i][j] > 0.1 || similarity[j][i] > 0.1) {
                if (!model_references[i]->isHard)
                    model_references[i]->isHard=true;
                if (!model_references[j]->isHard)
                    model_references[j]->isHard=true;
            }
        }

    }

    for (int i=0; i< model_references.size(); i++){
        if (model_references[i]->isHard)
            hardToTell.push_back(model_references[i]);
        else
            easyToTell.push_back(model_references[i]);
    }


#endif



#ifdef DEBUG
    auto now = std::chrono::high_resolution_clock::now();
    std::clock_t c_start = std::clock();

#endif

#ifndef USE_TEMPLATE
    ///preprocess the scene
    //scene->deBlur(true);
    scene -> build(alg);
#endif
    int approx_scale = scene->approximateScale();


#ifdef DEBUG
    std::clock_t c_end = std::clock();
    std::cerr << "[DEBUG] Scene:\t" << scene->path << "\n";
    std::cerr << "[DEBUG] Detected scale:\t" << approx_scale << "\n";
    std::cerr << "[DEBUG] Scene preprocessing CPU time: " << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << " ms\n";
#endif

#ifndef USE_TEMPLATE
    auto ghtmatcher = new GHTMatcher(scene);

    #ifdef FAST
    #ifdef DEBUG
        c_start = std::clock();
    #endif
    #ifdef USE_SIMILARITY
        auto multi = ghtmatcher->FastGHTMatch(model_references, alg, similarity, 0.25);
    #else
        auto multi = ghtmatcher->FastGHTMatch(model_references, alg);
    #endif
    #ifdef DEBUG
        c_end = std::clock();
        std::cerr << "[DEBUG] FastGHTMatch CPU time: " << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << " ms\n";
    #endif
    #else
        #ifdef DEBUG
            c_start = std::clock();
        #endif
    #ifdef USE_SIMILARITY
            ghtmatcher->FastGHTMatch(easyToTell, alg);
            auto multi = ghtmatcher->GHTMatch(hardToTell, alg);

    #else
        auto multi = ghtmatcher->GHTMatch(model_references, alg);
    #endif
        #ifdef  DEBUG
            c_end = std::clock();
            std::cerr << "[DEBUG] GHTMatch CPU time: " << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << " ms\n";
        #endif
    #endif
#else
    auto multi = templateMacth(model_references, scene);
#endif

#ifdef DEBUG

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - now;
    std::cerr << "[DEBUG] Execution completed in " << elapsed.count() << " seconds\n";

#endif
#ifndef USE_TEMPLATE
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

#else
//    std::vector<cv::Mat> hsvimg;
//    cv::Mat tmp;
//    colorscene.convertTo(tmp, cv::COLOR_BGR2HSV);
//    cv::split(tmp, hsvimg);
//    cv::imshow("HUE",hsvimg[0]);
//    cv::imshow("SAT",hsvimg[1]);
//    cv::imshow("VAL",hsvimg[2]);

    for (auto match : multi) {
        std::string modelname = match.first -> path;
        std::cout << "\nLooking for " << modelname << "...\n";
        for (int row = 0; row < match.second.rows; row++){
            for (int col = 0; col < match.second.cols; col++) {
                float t = match.second.at<float>(row, col);

                if (t > 0.999) {
                    //std::cout << "\tFound at (" << row << ", " << col << ")\t"<< t << "\n";

                    cv::putText(colorscene, ids[modelname], cv::Point2d(col, row),
                                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, CvScalar(250, 255,250));
                }
            }
        }
    }
    cv::imshow("Matches", colorscene);
    cv::waitKey(0);

#endif

}