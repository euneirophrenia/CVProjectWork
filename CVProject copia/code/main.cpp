//
// Created by Marco DiVi on 27/02/18.
//

#include "RichImage.h"

using namespace std;

/**
 * This is intended to be the main access to images, so that we can cache references.
 */

//TODO: create a ~factory to make it easy to select algorithm and parameters
int main(int argc, char** argv) {

    ///unify model sizes

    auto model = Images.getOrElse(context.BASE_PATH + "models/26.jpg", load(cv::IMREAD_GRAYSCALE));
    auto scene = Images.getOrElse(context.BASE_PATH + "scenes/e2.png", load(cv::IMREAD_GRAYSCALE));

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

    /*auto time = funcTime(strictDesciption, &model, detector, &matcher);
    cout << "Models processed in " << time << "ns\n";*/

    cv::Mat rescaled;

    //todo: make it so every image is scaled to more or less the same size (some are already "small")
    cv::resize(model.image, rescaled, model.image.size() / 4 );
    //todo: does it matter if i blur first and then resize? (the latter should be more conservative w.r.t. quality, imo)
    //blur(rescaled, model.image, CvSize(4,4));
    GaussianBlur(rescaled, model.image, CvSize(3,3), 4, 4);

    //todo: make it possible to create various combinations of extractor / evaluators
    //some cases may benefit from 1 algorithm keypoints and another one's evaluation
    //also, review the idea: model too close to each other may lead to the loss of some important features
    //strictDesciption(&model, detector, matcher, 0.2);

    detector-> detectAndCompute(model.image, cv::Mat(), model.keypoints, model.features);

    detector->detectAndCompute(scene.image, cv::Mat(), scene.keypoints, scene.features );
    auto matches = findModel(model.features, scene.features, matcher);

    cout << "Found " << matches.size() << " / " << context.MIN_MATCHES << " matches.\n";
    if (matches.size() > context.MIN_MATCHES) {

        showMatches(model, scene, matches);


    } else {
        cout << "Model " << model.path << " not found in " << scene.path << ".\n";
    }


}