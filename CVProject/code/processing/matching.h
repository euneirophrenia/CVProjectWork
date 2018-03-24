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



//todo: possibly move the threshold to the context
std::vector<cv::DMatch> findKnn(cv::Mat &modelFeatures, cv::Mat &targetFeatures, cv::DescriptorMatcher *matcher,
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
        float factor = 1.0f * minimum.height / image->image.size().height  ;
        cv::Mat rescaled;

        if (andBlur)
            GaussianBlur(image->image, rescaled, context.GAUSSIAN_KERNEL_SIZE, context.GAUSSIAN_X_SIGMA, context.GAUSSIAN_Y_SIGMA);
        else {
            rescaled = image->image;
        }

        cv::Size finalSize = CvSize(int(image->image.size().width * factor), int(image->image.size().height * factor));
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
            estimated_bary.x = scenept.x - scale * models[i]->houghModel[match.trainIdx][0];
            estimated_bary.y = scenept.y - scale * models[i]->houghModel[match.trainIdx][1];

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


std::vector<BlobPosition> GHTMatch(RichImage* model, RichImage* scene, Algorithm algo) {

    cv::Mat res = cv::Mat::zeros(scene->image.rows, scene->image.cols, CV_32S);

    if (scene->keypoints.empty())
        algo.detector->detectAndCompute(scene->image, cv::Mat(), scene->keypoints, scene->features);

    if (model->keypoints.empty())
        algo.detector->detectAndCompute(model->image, cv::Mat(), model->keypoints, model->features);


    std::vector<cv::DMatch> matches = findKnn(scene->features, model->features, algo.matcher, context["THRESHOLD"], true);


    for (auto match : matches) {
        double scale = scene->keypoints[match.queryIdx].size;
        double angle = scene->keypoints[match.queryIdx].angle - model->keypoints[match.trainIdx].angle;

        cv::Point2d scenept = scene->keypoints[match.queryIdx].pt;
        cv::Point2d estimated_bary;
        cv::Vec2d houghmodel = model->houghModel[match.trainIdx];


        houghmodel = rotate(houghmodel, angle);

        estimated_bary.x = (scenept.x +  scale * houghmodel[0]);
        estimated_bary.y = (scenept.y +  scale * houghmodel[1]);

        if (estimated_bary.x <0 || estimated_bary.y < 0 || estimated_bary.x > res.cols || estimated_bary.y > res.rows) {
            std::cerr << "Ignoring " << estimated_bary << "\n";
            //todo:: make it ~rubberband on the border, it may still provide useful insights
            continue;
        }

        for (int x= int(estimated_bary.x - scale/2); x<= int(estimated_bary.x + scale/2); x++) {
            for (int y= int(estimated_bary.y - scale/2); y<= int(estimated_bary.y + scale/2); y++) {
                res.at<int>(y,x) +=1;
            }
        }

    }

    cv::Mat labels(res.size(), CV_32S);
    int howmany = cv::connectedComponents(res > 0, labels, 8);

    if (howmany < 2)
        return std::vector<BlobPosition>();

    //std::cout << "Found " << howmany << " instances.\n";

    int current;
    BlobPosition blobs[howmany-1];

    for (int x= 0; x< labels.rows; x++) {
        for (int y= 0; y< labels.cols; y++) {
            current = labels.at<int>(x,y);

            ///background
            if (current == 0)
                continue;

            blobs[current-1].position += res.at<int>(x,y)*cv::Point2d(y,x);
            blobs[current-1].confidence += res.at<int>(x,y);
            blobs[current-1].area += 1;
        }
    }

    std::vector<BlobPosition> actual;

    for (int i=0; i<howmany-1; i++) {
        blobs[i].position /= blobs[i].confidence;
        blobs[i].confidence /= blobs[i].area;

        actual.push_back(blobs[i]);
        //std::cout << blobs[i].position << ", " << blobs[i].confidence << "\n";
    }

    return actual;
}


#endif //PROJECTWORK_MATCHING_H
