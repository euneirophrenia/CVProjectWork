//
// Created by Marco DiVi on 27/02/18.
//

#include "processing/matching.h"

#define TEST_SCENE "scenes/h2.jpg"
#define TEST_MODEL 1


//TODO: create a ~factory to make it easy to select algorithm and parameters
//todo: change some scripted params with the context ones


int main(int argc, char** argv){

    std::map<std::string, std::string> ids;

    auto msize = context.MODELS.size();

    for (int i =0; i< msize; i++) {
        ids[context.MODELS[i]] = fileName(context.MODELS[i], true);
    }

    /// setup the algorithm
    auto detector = cv::xfeatures2d::SIFT::create();
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


    uniform(model_references, true);

    for (auto m : model_references){
        m->build(alg, true);
    }

    //auto model = Images.getOrElse(context.BASE_PATH + "models/0.jpg", load(cv::IMREAD_GRAYSCALE));
    auto scene = new RichImage(context.BASE_PATH + TEST_SCENE); //Images.getOrElse(context.BASE_PATH + TEST_SCENE, load(cv::IMREAD_GRAYSCALE));
    scene -> build(alg);

    auto colorscene = cv::imread(context.BASE_PATH + TEST_SCENE, cv::IMREAD_COLOR);


    std::cout <<"Scene: " << scene->path << ":\n";


    /*auto time = funcTime(multiMatch, model_references, scene, detector, matcher);
    std::cout << "First scan done in " << time << "ns\n";

    time = funcTime(multiMatch, model_references, scene, detector, matcher);
    std::cout << "Second scan done in " << time << "ns\n";*/

    /*auto multi = multiMatch(model_references, scene, *alg);

    //todo:: try to match from the image to the model, instead of the contrary, since there are many instances of one model


    for (auto m : model_references) {
        if (multi[m].size() > context.MIN_MATCHES) {
            auto at = showMatches(*m, *scene, multi[m]);
            std::cout << "\tModel " << m->path << " found at " << at << " (" << multi[m].size() << "/" <<context.MIN_MATCHES << ")\n";
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



    auto multi = GHTMatch(model_references, scene, *alg);
    for (auto match : multi) {
        auto ghtmatch = match.second;
        std::string modelname = match.first;
        //auto ghtmatch = _ghtmatch(model_references[TEST_MODEL], scene, *alg);
        //std::cout << "Looking for " << model_references[TEST_MODEL]->path << "\n\n";
        std::cout << "\nLooking for " << modelname << "\n";

        if (!ghtmatch.empty()) {
            for (auto blob : ghtmatch) {
                std::cout << "\tFound at " << blob.position << "\t(conf: " << blob.confidence << ",\tarea:" << blob.area << ")\n";
                //cv::drawMarker(colorscene, blob.position, colors[blob.modelName]);

                cv::putText(colorscene, ids[blob.modelName], blob.position,
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, CvScalar(250, 255,250));

            }
        }
    }
    cv::imshow("Matches", colorscene);
    cv::waitKey(0);

}