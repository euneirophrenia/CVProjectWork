//
// Created by Marco DiVi on 27/02/18.
//

#include <vector>
#include "processing/matching.h"

#define TEST_SCENE "scenes/e1.png"


//TODO: create a ~factory to make it easy to select algorithm and parameters
//todo: change some scripted params with the context ones


int main(int argc, char** argv) {

    /// unify model sizes
    std::vector<RichImage*> model_references;
    for (auto p : context.MODELS) {
        model_references.push_back(Images.getOrElse(p, load(cv::IMREAD_GRAYSCALE)));
    }

    uniform(model_references);

    //auto model = Images.getOrElse(context.BASE_PATH + "models/0.jpg", load(cv::IMREAD_GRAYSCALE));
    auto scene = Images.getOrElse(context.BASE_PATH + TEST_SCENE, load(cv::IMREAD_GRAYSCALE));

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


    /*auto time = funcTime(multiMatch, model_references, scene, detector, matcher);
    std::cout << "First scan done in " << time << "ns\n";

    time = funcTime(multiMatch, model_references, scene, detector, matcher);
    std::cout << "Second scan done in " << time << "ns\n";*/

    auto multi = multiMatch(model_references, scene, detector, matcher);

    std::cout <<"Scene: " << scene->path << ":\n";
    for (auto m : model_references) {
        if (multi[m].size() > context.MIN_MATCHES) {
            auto at = showMatches(*m, *scene, multi[m]);
            std::cout << "\tModel " << m->path << " found at " << at << " (" << multi[m].size() << "/" <<context.MIN_MATCHES << ")\n";
        }
    }

    /*detector-> detectAndCompute(model->image, cv::Mat(), model->keypoints, model->features);

    detector->detectAndCompute(scene->image, cv::Mat(), scene->keypoints, scene->features );
    auto matches = findModel(model->features, scene->features, matcher);

    cout << "Found " << matches.size() << " / " << context.MIN_MATCHES << " matches.\n";
    if (matches.size() > context.MIN_MATCHES) {

        showMatches(*model, *scene, matches);


    } else {
        std::cout << "Model " << model->path << " not found in " << scene->path << ".\n";
    }*/


}