//
// Created by Marco DiVi on 07/03/18.
//

/**
 * This file gathers all the functions used to perform matches and operations on them
 */

#ifndef PROJECTWORK_MATCHING_H
#define PROJECTWORK_MATCHING_H

#include "RichImage.h"
#include <vector>
#include <string>
#include "blob.h"


std::vector<cv::DMatch> findKnn(cv::Mat &targetFeatures, cv::Mat &modelFeatures, cv::DescriptorMatcher *matcher,
                                float threshold = 0.7, bool andFilter=true) {

    std::vector<std::vector<cv::DMatch>> matches;
    matcher->knnMatch(modelFeatures, targetFeatures, matches, 2);

    std::vector<cv::DMatch> goodMatches;

    if (andFilter){
        for (int k = 0; k < matches.size(); k++)
        {
            if ((matches[k][0].distance < threshold *(matches[k][1].distance)) && (matches[k].size() <= 2 && matches[k].size()>0) ) {
                goodMatches.push_back(matches[k][0]);
            }
        }
    }
    else {
        for (int k = 0; k < matches.size(); k++)
                goodMatches.push_back(matches[k][0]);
    }
    return goodMatches;
}

std::vector<cv::DMatch> MultiFindKnn(std::vector<cv::Mat> modelFeatures, cv::Mat& targetFeatures, cv::DescriptorMatcher* matcher,
                                                float threshold = 0.6, bool andClear = true) {
    std::vector<std::vector<cv::DMatch>> matches;

    matcher->add(modelFeatures);
    matcher->knnMatch(targetFeatures, matches, 2);

    std::vector<cv::DMatch> goodMatches;
    int dirtytrick;

    for (int k = 0; k < matches.size(); k++)
    {
        if ((matches[k][0].distance < threshold *(matches[k][1].distance)) && (matches[k].size() <= 2 && matches[k].size()>0) ) {
            dirtytrick = matches[k][0].queryIdx;
            matches[k][0].queryIdx = matches[k][0].trainIdx;
            matches[k][0].trainIdx = dirtytrick;
            goodMatches.push_back(matches[k][0]);
        }
    }

    if (andClear)
        matcher->clear();

    return goodMatches;

}

std::vector<cv::DMatch> findRadius(cv::Mat &modelFeatures, cv::Mat &targetFeatures, cv::DescriptorMatcher * matcher, float distance){
    std::vector<std::vector<cv::DMatch>> matches;
    matcher->radiusMatch(modelFeatures, targetFeatures, matches, distance);

    std::vector<cv::DMatch> goodMatches;

    for (int k = 0; k < matches.size(); k++) {
        if (matches[k].size() > 0)
            goodMatches.push_back(matches[k][0]);
    }
    return goodMatches;
}

cv::Rect boundingRect(RichImage* model, RichImage* sceneImage, std::vector<cv::DMatch> good_matches){
    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( model->keypoints[ good_matches[i].queryIdx ].pt );
        scene.push_back( sceneImage->keypoints[ good_matches[i].trainIdx ].pt );
    }

    cv::Mat H = findHomography( obj, scene, CV_RANSAC, 3);

    if (H.empty())
        return cv::Rect2d();

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( model->image.cols, 0 );
    obj_corners[2] = cvPoint( model->image.cols, model->image.rows ); obj_corners[3] = cvPoint( 0, model->image.rows );
    std::vector<cv::Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);
    return cv::boundingRect(scene_corners);

}

cv::Point2d localizeMatches(RichImage model, RichImage sceneImage, std::vector<cv::DMatch> good_matches,
                            cv::Scalar withColor = cv::Scalar(0, 255, 0),
                            bool debugShow = false) {
    cv::Mat img_matches;

    Context& ctx = Context::getInstance();

    if (debugShow) {
        drawMatches(model.image, model.keypoints, sceneImage.image, sceneImage.keypoints,
                    good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


        cv::imshow("matches", img_matches);
        cv::waitKey(0);
    }

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

    if (debugShow) {
        auto color = cv::imread(ctx.BASE_PATH + sceneImage.path, cv::IMREAD_COLOR);
        cv::rectangle(color, rect, withColor, 2, cv::LINE_AA);
        cv::imshow(model.path + " against " + sceneImage.path, color);
        cv::waitKey(0);
    }
    return 0.5*(rect.br() + rect.tl());
}

/// unify all models size and blur if needed, so that more or less everything is at the same resolution
void uniform(std::vector<RichImage*> models,  int approximate_scale = -1) {

    cv::Size minimum;
    if (approximate_scale < 0) {
        minimum = models[0]->image.size();

        for (auto image : models) {
            if (image->image.size().area() < minimum.area())
                minimum = image->image.size();
        }
    }

    else {
        minimum.height = approximate_scale;
    }

    /// rescale each image so that its height matches the minimum one
    // I chose height arbitrarily, I could have chosen the width, just not both, i want to preserve proportions
    for (auto image : models) {
        float factor = 1.0f * minimum.height / image->image.size().height  ;
        cv::Mat rescaled;

        if (factor < 1)
            GaussianBlur(image->image, rescaled, context.GAUSSIAN_KERNEL_SIZE, context.GAUSSIAN_X_SIGMA, context.GAUSSIAN_Y_SIGMA);
        else {
            rescaled = image->image;
        }

        cv::Size finalSize = CvSize(int(image->image.size().width * factor), int(image->image.size().height * factor));
        cv::resize(rescaled, image->image, finalSize);

    }

}

std::map<RichImage*, std::vector<cv::DMatch>> multiMatch(std::vector<RichImage*> models, RichImage* target,
                                                         Algorithm algo, bool fast=false){

    std::map<RichImage*, std::vector<cv::DMatch>> res;

    if (target->features.empty() || target->keypoints.empty())
        algo.detector->detectAndCompute(target->image, cv::Mat(), target->keypoints, target->features );

    std::vector<cv::Mat> allfeats;

    if (fast) {
        for (auto model : models) {
            allfeats.push_back(model->features);
        }

        std::vector<cv::DMatch> all = MultiFindKnn(allfeats, target->features, algo.matcher,
                                                   context.GOOD_MATCH_RATIO_THRESHOLD);

        for (auto m : all) {
            res[models[m.imgIdx]].push_back(m);
        }
        return res;
    }

    u_long totalMatches[models.size()];

    /// find for each model their respective matches
    cv::DMatch matches[target->keypoints.size()][models.size()];
    for (int i=0; i<models.size(); i++) {
        auto model = models[i];
        if (model->keypoints.empty() || model->features.empty())
            algo.detector->detectAndCompute(model->image, cv::Mat(), model->keypoints, model->features);

        std::vector<cv::DMatch> localmatches = findKnn(model->features, target->features, algo.matcher,
                                                       context.GOOD_MATCH_RATIO_THRESHOLD);
        totalMatches[i] = localmatches.size();
        for (auto match : localmatches) {
            matches[match.trainIdx][i] = match;
        }
    }

    /// find, for each keypoint in the image, the best match: if a keypoint matched for more than 1 model, keep the model with more matches overall
    // maybe improve to keep the best looking match (highest ratio to the second nearest,e.g.)
    for (int keypoint = 0; keypoint < target->keypoints.size(); keypoint ++) {
        int best_model = -1;
        for (int model = 0; model < models.size(); model++) {
            if (matches[keypoint][model].trainIdx >= 0 &&
                (best_model < 0 || totalMatches[best_model] < totalMatches[model]) &&
                totalMatches[model] >= context.MIN_MATCHES) {
                best_model = model;
            }
        }
        if (best_model >= 0) {
            res[models[best_model]].push_back(matches[keypoint][best_model]);
        }
    }

    return res;
}



#endif //PROJECTWORK_MATCHING_H
