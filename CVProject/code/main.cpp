//
// Created by Marco DiVi on 27/02/18.
//

#include "processing/matching.h"
#include <vector>

using namespace std;

#define TEST_SCENE "scenes/e1.png"

/// unify all models size and blur, so that more or less everything is at the same resolution

//TODO: create a ~factory to make it easy to select algorithm and parameters
int main(int argc, char** argv) {

    /// unify model sizes
    vector<RichImage*> model_references;
    for (auto p : context.MODELS) {
        model_references.push_back(Images.getOrElse(p, load(cv::IMREAD_GRAYSCALE)));
    }

    uniform(model_references);

    //auto model = Images.getOrElse(context.BASE_PATH + "models/0.jpg", load(cv::IMREAD_GRAYSCALE));
    auto scene = Images.getOrElse(context.BASE_PATH + TEST_SCENE, load(cv::IMREAD_GRAYSCALE));

    auto detector = cv::xfeatures2d::SIFT::create();
    //auto detector = cv::xfeatures2d::SURF::create(400);
    //auto detector = cv::BRISK::create(60, 4, 1.0f);
    //auto detector = cv::ORB::create();

    /// For SIFT / SURF
     auto indexparams = new cv::flann::KDTreeIndexParams(5);

    /// FOR BRISK / ORB
    //auto indexparams = new cv::flann::LshIndexParams(20,10,2);


    cv::flann::SearchParams* searchParams = new cv::flann::SearchParams(50);
    cv::FlannBasedMatcher* matcher = new cv::FlannBasedMatcher(indexparams, searchParams);


    /*auto time = funcTime(multiMatch, model_references, scene, detector, matcher);
    cout << "First scan done in " << time << "ns\n";

    time = funcTime(multiMatch, model_references, scene, detector, matcher);
    cout << "Second scan done in " << time << "ns\n";*/

    auto multi = multiMatch(model_references, scene, detector, matcher);

    for (auto m : model_references) {
        if (multi[m].size() > context.MIN_MATCHES) {
            showMatches(*m, *scene, multi[m]);
        }
        else {
            cout << "Model " << m->path << " not found in " << scene->path << " (" << multi[m].size() << "/" <<context.MIN_MATCHES << ")\n";
        }
    }

    /*detector-> detectAndCompute(model->image, cv::Mat(), model->keypoints, model->features);

    detector->detectAndCompute(scene->image, cv::Mat(), scene->keypoints, scene->features );
    auto matches = findModel(model->features, scene->features, matcher);

    cout << "Found " << matches.size() << " / " << context.MIN_MATCHES << " matches.\n";
    if (matches.size() > context.MIN_MATCHES) {

        showMatches(*model, *scene, matches);


    } else {
        cout << "Model " << model->path << " not found in " << scene->path << ".\n";
    }*/


}