//
// Created by Marco DiVi on 27/02/18.
//

#include "processing/matching.h"
#include <chrono>

//#include "processing/InfiniteMatrix.h"

#define TEST_SCENE "scenes/m3.png"
//#define TEST_MODEL 1



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

    /// unify model sizes
    std::vector<RichImage*> model_references;
    for (auto& p : context.MODELS) {
        model_references.push_back(Images.getOrElse(p, load(cv::IMREAD_GRAYSCALE)));
    }

    auto scene = new RichImage(context.BASE_PATH + TEST_SCENE); //Images.getOrElse(context.BASE_PATH + TEST_SCENE, load(cv::IMREAD_GRAYSCALE));
    auto colorscene = cv::imread(context.BASE_PATH + TEST_SCENE, cv::IMREAD_COLOR);

#ifdef DEBUG
    auto now = std::chrono::high_resolution_clock::now();
#endif

    scene->deBlur(false);

    scene -> build(alg);
    int approx_scale = scene->approximateScale();

#ifdef DEBUG
    std::cerr << "[DEBUG] Scene:\t" << scene->path << "\n";
    std::cerr << "[DEBUG] Detected scale:\t" << approx_scale << "\n";
#endif

    uniform(model_references, true, approx_scale); //also smoothing
    //uniform(model_references, false, approx_scale);  // without smoothing

    for (auto m : model_references){
        //m->deBlur();
        m->build(alg, true);
    }

     //auto model = Images.getOrElse(context.BASE_PATH + "models/0.jpg", load(cv::IMREAD_GRAYSCALE));



    //std::cout <<"Scene: " << scene->path << ":\n";



    /*auto multim = multiMatch(model_references, scene, *alg, false);


    for (auto m : model_references) {
        if (multim[m].size() > context.MIN_MATCHES) {
            auto at = localizeMatches(*m, *scene, multim[m], CvScalar(0,255,0), false);
            std::cout << "\tModel " << m->path << " found at " << at << " (" << multim[m].size() << "/" <<context.MIN_MATCHES << ")\n";
        }
    }*/


    /*auto testmodel = new RichImage("../CVProject/models/6.jpg");
    cv::Mat testcolor = cv::imread("../CVProject/models/6-1.jpg", cv::IMREAD_COLOR);
    auto testrotated = new RichImage("../CVProject/models/6-1.jpg");

    testmodel->build(alg, true);
    testrotated->build(alg);

    auto ghttest = _ghtmatch(testmodel, testrotated, *alg);
    for (auto blob : ghttest) {
        if (blob.confidence >= context["MIN_HOUGH_VOTES"]) {
            std::cout << "\tFound at " << blob.position << "\t(conf: " << blob.confidence << ",\tarea:"
                                                 << blob.area << ")\n";
            cv::drawMarker(testcolor, blob.position, CvScalar(255, 255, 255));
        }
    }

    cv::imshow("test hough", testcolor);
    cv::waitKey(0);

    exit(1);*/

    //auto multi = GHTMatch(model_references, scene, *alg);
    auto multi = FastGHTMatch(model_references, scene, *alg);

#ifdef DEBUG
    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - now;
    std::cerr << "[DEBUG] Execution completed in " << elapsed.count() << " seconds \n";

#endif
    for (auto match : multi) {
        auto ghtmatch = match.second;
        std::string modelname = match.first -> path;
        std::cout << "\nLooking for " << modelname << "...\n";

        if (!ghtmatch.empty()) {
            for (auto blob : ghtmatch) {
                std::cout << "\tFound at " << blob.position << "\t(conf: " << blob.confidence << ",\tarea: " << blob.area << ")\n";
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