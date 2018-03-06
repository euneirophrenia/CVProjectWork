//
// Created by Marco DiVi on 03/03/18.
//

/**
 * This module collects some useful / recurrent functions I needed throughout the project.
 */

//TODO:: test the strictDescription, optimize it and complete translation of the python work

#ifndef TOOLS_H
#define TOOLS_H

#include "ResourcePool.h"
#include "tools.h"
#include "context.h"

#include "opencv2/xfeatures2d/nonfree.hpp"

const Context& context = Context::getInstance();

struct RichImage {
    std::string path;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat features;

    operator cv::Mat() const {
        return image;
    }

    operator cv::InputArray () const {
        return cv::InputArray(image);
    }

    void show(std::string windowname, int waitkey=0) {
        cv::imshow(windowname, image);
        cv::waitKey(waitkey);
    }
};



std::function<RichImage(std::string)> load(int flag) {
    return [=](std::string filename) {
        RichImage res;
        res.image = cv::imread(filename, flag);
        if (res.image.empty()){
            throw std::invalid_argument("Image not loaded properly " + filename);
        }
        res.path = filename;
        return res;

    };
}


/**
 * Global entry point for caching images, read them only once and have them forever.
 * Might not be a good idea, but it could be used to ensure that every image is loaded in the same way or preprocessed
 * properly. Its use is not mandatory. I think i will expand it by caching "Rich" images.
 * I.e images with also already computed features and keypoints so that we don't need to repeat that across various scenes.
 */
ResourcePool<std::string, RichImage> Images;


//todo: possibly move the threshold to the context
std::vector<cv::DMatch> findModel(cv::Mat modelFeatures, cv::Mat targetFeatures, cv::DescriptorMatcher* matcher,
                                  float threshold = 0.7) {

    std::vector<std::vector<cv::DMatch>> matches;
    matcher->knnMatch(modelFeatures, targetFeatures, matches, 2);


    std::vector<cv::DMatch> goodMatches;

    for (int k = 0; k < matches.size(); k++)
    {
        if ( (matches[k][0].distance < threshold *(matches[k][1].distance)) &&
             ((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
        {
            // take the first result only if its distance is smaller than 0.6*second_best_dist
            // that means this descriptor is ignored if the second distance is bigger or of similar
            goodMatches.push_back( matches[k][0] );
        }
    }
    return goodMatches;
}

std::map<RichImage,std::vector<cv::DMatch>> findModels(std::vector<std::string> models, std::string target,
                                                       cv::FeatureDetector* detector, cv::DescriptorMatcher* matcher, float threshold=0.7)
{
    std::map<RichImage, std::vector<cv::DMatch>> res;
    for (auto m : models) {
        //if m does not have feature computed -> compute them
        //find m in target (findModel)
        //for each goodmatch of model m, check if there's already another model matching better at that location
        //if so, remove that match
        //pray
    }

    return res;
}


//TODO: possibly, delete matches symmetrically, also on other models
// that may lead to some inconsistencies, though. 2 seconds for 6 models is already nice IMO, since this operation can be carried out offline
/**
 * Instead of only computing the model description, also forget those feature that would erroneously match against other models.
 * @param model the RichImage from which to extract keypoints and features.
 * @param detector the keypoint detector to use (passed by reference)
 * @param matcher the matcher to use to compare against the other models (by reference)
 */
void strictDesciption(RichImage* model, cv::FeatureDetector* detector, cv::DescriptorMatcher* matcher, float threshold = 0.4) {
    if (model->keypoints.empty() || model->features.empty())
        /// extract features and keypoints from <model>
        detector->detectAndCompute(model->image, cv::Mat(), model->keypoints, model->features);


    /// fetch all other models
    std::vector<RichImage> othermodels;

    for(int i=0; i < context.MODELS.size(); i++) {
        if (context.MODELS[i] != context.BASE_PATH + model->path ) {
            auto other = Images.getOrElse(context.MODELS[i], load(cv::IMREAD_GRAYSCALE));
            othermodels.push_back(*other);
        }
    }

    /// for each other model <other>:
    for (int i=0; i < othermodels.size(); i++) {
        std::cout<<"Processing "<<model->path <<" against "<<othermodels[i].path<<"\n";
        ///        extract features from <other>, if needed
        if (othermodels[i].keypoints.empty() || othermodels[i].features.empty())
            detector -> detectAndCompute(othermodels[i].image, cv::Mat(), othermodels[i].keypoints, othermodels[i].features);

        ///        match against <model>, only keep the goodmatches
        std::vector<cv::DMatch> matches = findModel(model->features, othermodels[i].features, matcher, threshold);

        std::vector<size_t> toRemove;

        /// for each match found we must eliminate the conflicting matches
        for(int j=0; j< matches.size(); j++) {
             // if there was a function to also delete 1 single row of the matrix without creating 13838292 other mats,
             //then i'd do everything inside this loop, instead of saving the indices to delete and only after actually erasing them.
             //model->keypoints.erase(model->keypoints.begin() + matches[j].queryIdx);
             toRemove.push_back(matches[j].queryIdx);

        }
        model->keypoints = erase_indices(model->keypoints, toRemove);
        model->features = erase_rows(model->features, toRemove);
    }
}

void showMatches(RichImage model, RichImage sceneImage, std::vector<cv::DMatch> good_matches, cv::Scalar withColor = cv::Scalar(0,255,0)) {
    cv::Mat img_matches;

    Context& ctx = Context::getInstance();

    drawMatches( model.image, model.keypoints, sceneImage.image, sceneImage.keypoints,
                 good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::imshow("matches", img_matches);
    cv::waitKey(0);

    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( model.keypoints[ good_matches[i].queryIdx ].pt );
        scene.push_back( sceneImage.keypoints[ good_matches[i].trainIdx ].pt );
    }

    cv::Mat H = findHomography( obj, scene, CV_RANSAC, 5.0 );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( model.image.cols, 0 );
    obj_corners[2] = cvPoint( model.image.cols, model.image.rows ); obj_corners[3] = cvPoint( 0, model.image.rows );
    std::vector<cv::Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);
    auto rect = cv::boundingRect(scene_corners);

    auto color = cv::imread(ctx.BASE_PATH+sceneImage.path, cv::IMREAD_COLOR);
    cv::rectangle(color, rect, withColor, 2, cv::LINE_AA);

    cv::imshow(model.path + " against " + sceneImage.path, color);

    cv::waitKey(0);
}

#endif //TOOLS_H
