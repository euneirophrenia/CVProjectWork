//
// Created by Marco DiVi on 27/02/18.
//

#include "RichImage.h"

using namespace std;


/// unify all models size and blur, so that more or less everything is at the same resolution

void uniform(vector<RichImage*> models) {

    cv::Size minimum = models[0]->image.size();

    for (auto image : models) {
        if (image->image.size().area() < minimum.area())
            minimum = image->image.size();
    }

    /// rescale each image so that its height matches the minimum one
    // I chose height arbitrarily, I could have chosen the width, just not both, i want to preserve proportions
    for (auto image : models) {
        float factor = 1.0f * image->image.size().height / minimum.height ;
        cv::Mat rescaled;

        cv::Size finalSize = CvSize(int(image->image.size().width / factor), int(image->image.size().height / factor));
        cv::resize(image->image, rescaled, finalSize);

        //todo: does it matter if i blur first and then resize? (the latter should be more conservative w.r.t. quality, imo)
        GaussianBlur(rescaled, image->image, context.GAUSSIAN_KERNEL_SIZE, context.GAUSSIAN_X_SIGMA, context.GAUSSIAN_Y_SIGMA);
    }

}

//TODO: create a ~factory to make it easy to select algorithm and parameters
int main(int argc, char** argv) {

    /// unify model sizes
    vector<RichImage*> model_references;
    for (auto p : context.MODELS) {
        model_references.push_back(Images.getOrElse(p, load(cv::IMREAD_GRAYSCALE)));
    }

    uniform(model_references);

    auto model = Images.getOrElse(context.BASE_PATH + "models/0.jpg", load(cv::IMREAD_GRAYSCALE));
    auto scene = Images.getOrElse(context.BASE_PATH + "scenes/e1.png", load(cv::IMREAD_GRAYSCALE));


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

    //todo: make it possible to create various combinations of extractor / evaluators
    //some cases may benefit from 1 algorithm keypoints and another one's evaluation
    //also, review the idea: model too close to each other may lead to the loss of some important features
    //strictDesciption(&model, detector, matcher, 0.2);

    detector-> detectAndCompute(model->image, cv::Mat(), model->keypoints, model->features);

    detector->detectAndCompute(scene->image, cv::Mat(), scene->keypoints, scene->features );
    auto matches = findModel(model->features, scene->features, matcher);

    cout << "Found " << matches.size() << " / " << context.MIN_MATCHES << " matches.\n";
    if (matches.size() > context.MIN_MATCHES) {

        showMatches(*model, *scene, matches);


    } else {
        cout << "Model " << model->path << " not found in " << scene->path << ".\n";
    }


}