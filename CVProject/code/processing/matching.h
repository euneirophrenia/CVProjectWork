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



//todo: possibly move the threshold to the context
std::vector<cv::DMatch> findKnn(cv::Mat &modelFeatures, cv::Mat &targetFeatures, cv::DescriptorMatcher *matcher,
                                float threshold = 0.7) {

    std::vector<std::vector<cv::DMatch>> matches;
    matcher->knnMatch(modelFeatures, targetFeatures, matches, 2);

    std::vector<cv::DMatch> goodMatches;


    for (int k = 0; k < matches.size(); k++)
    {
        if ( (matches[k][0].distance < threshold *(matches[k][1].distance)) && (matches[k].size() <= 2 && matches[k].size()>0) ) {
            goodMatches.push_back(matches[k][0]);
        }
    }
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

cv::Point2d showMatches(RichImage model, RichImage sceneImage, std::vector<cv::DMatch> good_matches, cv::Scalar withColor = cv::Scalar(0,255,0)) {
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
    return 0.5*(rect.br() + rect.tl());
}

/// unify all models size and blur, so that more or less everything is at the same resolution
void uniform(std::vector<RichImage*> models, bool andBlur=false) {

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

        if (andBlur)
            GaussianBlur(image->image, rescaled, context.GAUSSIAN_KERNEL_SIZE, context.GAUSSIAN_X_SIGMA, context.GAUSSIAN_Y_SIGMA);
        else {
            rescaled = image->image;
        }

        //todo:: find a better estimate of the size
        cv::Size finalSize = CvSize(int(image->image.size().width / factor), int(image->image.size().height / factor));
        cv::resize(rescaled, image->image, finalSize);

    }

}

std::map<RichImage*, std::vector<cv::DMatch>> multiMatch(std::vector<RichImage*> models, RichImage* target, Algorithm algo){

    std::map<RichImage*, std::vector<cv::DMatch>> res;

    if (target->features.empty() || target->keypoints.empty())
        algo.detector->detectAndCompute(target->image, cv::Mat(), target->keypoints, target->features );

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
        for (int model = 0; model < models.size(); model++){
            if (matches[keypoint][model].trainIdx >= 0 && (best_model<0 || totalMatches[best_model] < totalMatches[model]) &&
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

std::map<RichImage*, std::vector<std::vector<cv::DMatch>>> GHTmultiMatch(std::vector<RichImage*> models, RichImage* scene, Algorithm algo) {

    std::map<RichImage*, std::vector<std::vector<cv::DMatch>>> res;

    if (scene->keypoints.empty())
        algo.detector->detectAndCompute(scene->image, cv::Mat(), scene->keypoints, scene->features);

    for (int i=0; i<models.size(); i++){

        std::vector<cv::DMatch> votes[scene->image.rows][scene->image.cols];


        std::vector<cv::DMatch> matches = findKnn(models[i]->features, scene->features, algo.matcher,
                                                  context["THRESHOLD"]);

        for (auto match : matches) {
            double scale = scene->keypoints[match.queryIdx].size / models[i]->keypoints[match.trainIdx].size;
            cv::Point2d scenept = scene->keypoints[match.queryIdx].pt;
            cv::Point2d estimated_bary;
            estimated_bary.x = scenept.x - scale * (*models[i]->houghModel)[match.trainIdx][0];
            estimated_bary.y = scenept.y - scale * (*models[i]->houghModel)[match.trainIdx][1];

            votes[(int)estimated_bary.x][(int)estimated_bary.y].push_back(match);
        }

        res[models[i]] = std::vector<std::vector<cv::DMatch>>();

        for (int j=0; j< scene->image.rows; j++) {
            for (int k =0; k < scene->image.cols; k++) {
                if (votes[j][k].size() > context["MIN_HOUGH_VOTES"])
                    res[models[i]].push_back(votes[j][k]);
            }
        }
    }

    return res;
}

//todo:: the houghModel probably gets built wrongly, look into that
// also, find a way to collapse very close votes
cv::Mat GHTMatch(RichImage* model, RichImage* scene, Algorithm algo) {

    cv::Mat res = cv::Mat::zeros(scene->image.rows, scene->image.cols, CV_32SC1);

    if (scene->keypoints.empty())
        algo.detector->detectAndCompute(scene->image, cv::Mat(), scene->keypoints, scene->features);

    if (model->keypoints.empty())
        algo.detector->detectAndCompute(model->image, cv::Mat(), model->keypoints, model->features);


    std::vector<cv::DMatch> matches = findKnn(scene->features, model->features, algo.matcher, context["THRESHOLD"]);

    std::ofstream data;
    data.open("/Users/marcodivincenzo/Documents/Ingegneria/Magistrale/CV/Python/ProjectWork/data.txt", std::ios::out);

    for (auto match : matches) {
        double scale = scene->keypoints[match.queryIdx].size;
        double angle = scene->keypoints[match.queryIdx].angle - model->keypoints[match.trainIdx].angle;

        cv::Point2d scenept = scene->keypoints[match.queryIdx].pt;
        cv::Point2d estimated_bary;
        cv::Vec2d houghmodel = (*model->houghModel)[match.trainIdx];

        /*std::cout << "Size :" << model->keypoints[match.trainIdx].size << "/" << scene->keypoints[match.queryIdx].size << "\n";
        std::cout << "Angle :" << model->keypoints[match.trainIdx].angle << "/" << scene->keypoints[match.queryIdx].angle << "\n";
        std::cout << "Response :" << model->keypoints[match.trainIdx].response << "/" << scene->keypoints[match.queryIdx].response << "\n";
        std::cout << "------------\n";*/

        houghmodel = rotate(houghmodel, angle);

        estimated_bary.x = (scenept.x +  scale * houghmodel[0]);
        estimated_bary.y = (scenept.y +  scale * houghmodel[1]);

        if (estimated_bary.x <0 || estimated_bary.y < 0 || estimated_bary.x > res.cols || estimated_bary.y > res.rows) {
            std::cerr << "Ignoring " << estimated_bary << "\n";
            continue;
        }

        data << estimated_bary.x << "," << estimated_bary.y << "\n";
        res.at<int>(estimated_bary) += 1;
    }
    data.close();

    //todo:: blob analysis, merge the ~connected estimated barys and provide only one (maybe with a measure of connectiveness
    //filter here spurious matches by either:
    //      - setting a threshold
    //      - tuning search parameters to be strict -> may break connectiveness

    //todo:: finally, since we're basically ignoring perfect matches, test with less powerful detectors, to speed up processing

    return res;
}


#endif //PROJECTWORK_MATCHING_H
