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
                                float threshold = 0.7, bool andFilter=true, bool doDirtyTrick = false) {

    std::vector<std::vector<cv::DMatch>> matches;
    matcher->knnMatch(modelFeatures, targetFeatures, matches, 2);  //from the model to the target, use accordingly
    std::vector<cv::DMatch> goodMatches;

    if (andFilter){
        for (int k = 0; k < matches.size(); k++)
        {
            if ((matches[k][0].distance < threshold *(matches[k][1].distance)) && (matches[k].size() <= 2 && matches[k].size()>0) ) {
				if (doDirtyTrick){
					int dirtytrick = matches[k][0].queryIdx;
					matches[k][0].queryIdx = matches[k][0].trainIdx;
					matches[k][0].trainIdx = dirtytrick;
				}
                goodMatches.push_back(matches[k][0]);
            }
        }
    }
    else {
        for (int k = 0; k < matches.size(); k++) {
			if (doDirtyTrick){
				int dirtytrick = matches[k][0].queryIdx;
				matches[k][0].queryIdx = matches[k][0].trainIdx;
				matches[k][0].trainIdx = dirtytrick;
			}
			goodMatches.push_back(matches[k][0]);
		}
    }
    return goodMatches;
}

//std::vector<cv::DMatch> findKnn(cv::Mat &targetFeatures, cv::Mat &modelFeatures, cv::DescriptorMatcher *matcher,
//								float threshold = 0.7, bool andFilter=true, bool doDirtyTrick = false) {
//
//	std::vector<std::vector<cv::DMatch>> matches;
//	matcher->knnMatch(targetFeatures, modelFeatures, matches, 2);  //from the scene to the model
//	std::vector<cv::DMatch> goodMatches;
//
//	if (andFilter){
//		for (int k = 0; k < matches.size(); k++)
//		{
//			if ((matches[k][0].distance < threshold *(matches[k][1].distance)) && (matches[k].size() <= 2 && matches[k].size()>0) ) {
//				if (doDirtyTrick){
//					int dirtytrick = matches[k][0].queryIdx;
//					matches[k][0].queryIdx = matches[k][0].trainIdx;
//					matches[k][0].trainIdx = dirtytrick;
//				}
//				goodMatches.push_back(matches[k][0]);
//			}
//		}
//	}
//	else {
//		for (int k = 0; k < matches.size(); k++) {
//			if (doDirtyTrick){
//				int dirtytrick = matches[k][0].queryIdx;
//				matches[k][0].queryIdx = matches[k][0].trainIdx;
//				matches[k][0].trainIdx = dirtytrick;
//			}
//			goodMatches.push_back(matches[k][0]);
//		}
//	}
//	return goodMatches;
//}


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

std::vector<cv::DMatch> MultiFindKnnWithSimilarity(std::vector<cv::Mat> modelFeatures, cv::Mat& targetFeatures, cv::DescriptorMatcher* matcher,
                                                   float** similarity, float simThreshold = 0.6,float ratioThreshold = 0.6, bool andClear = true) {

    std::vector<std::vector<cv::DMatch>> matches;

    matcher->add(modelFeatures);
    matcher->knnMatch(targetFeatures, matches, 2);

    std::vector<cv::DMatch> goodMatches;
    int dirtytrick;

    for (int k = 0; k < matches.size(); k++)
    {

        if (matches[k][0].distance < ratioThreshold *(matches[k][1].distance) || similarity[matches[k][0].imgIdx][matches[k][1].imgIdx] >= simThreshold) {
            dirtytrick = matches[k][0].queryIdx;
            matches[k][0].queryIdx = matches[k][0].trainIdx;
            matches[k][0].trainIdx = dirtytrick;
            goodMatches.push_back(matches[k][0]);
        }

        if (similarity[matches[k][0].imgIdx][matches[k][1].imgIdx] >= simThreshold) {
            dirtytrick = matches[k][1].queryIdx;
            matches[k][1].queryIdx = matches[k][1].trainIdx;
            matches[k][1].trainIdx = dirtytrick;
            goodMatches.push_back(matches[k][1]);
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

    cv::Mat H = findHomography(obj, scene, CV_RANSAC, 3);

    if (H.empty())
        return cv::Rect2d();

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( model->image.cols, 0 );
    obj_corners[2] = cvPoint( model->image.cols, model->image.rows );
    obj_corners[3] = cvPoint( 0, model->image.rows );
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
            GaussianBlur(image->image, rescaled, context.GAUSSIAN_KERNEL_SIZE, context.GAUSSIAN_X_SIGMA/ factor, context.GAUSSIAN_Y_SIGMA / factor );
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

    std::vector<cv::Mat> allfeats;

    if (fast) { ///EASY MODELS
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

	///HARD MODELS

    u_long totalMatches[models.size()];
	cv::Point positions[models.size()];
	std::vector<cv::DMatch> matches[models.size()];
	bool isAlive[models.size()];


	/// find for each model their respective matches
    for (int i=0; i<models.size(); i++) {
        auto model = models[i];

        std::vector<cv::DMatch> localmatches = findKnn(model->features, target->features, algo.matcher,
                                                       context["THRESHOLD"], true, true);
        totalMatches[i] = localmatches.size();
        isAlive[i] = totalMatches[i] > context.MIN_MATCHES; //is a potential good candidate?
        if (isAlive[i]) {
            positions[i] = localizeMatches(*model, *target, localmatches);
            matches[i] = localmatches;
        }
        else {
            Logger::log("Ignoring " + model->path + "\t(" +  std::to_string(totalMatches[i]) + " / " + std::to_string(context.MIN_MATCHES) + ")");
            positions[i] = cv::Point2d(-1, -1);
            matches[i] = std::vector<cv::DMatch>();
        }

    }

    ///solve conflicts, if two models have been localized too close to each other, keep the one with more evidence
    //(most number of overall matches)
	for (int i=0; i< models.size(); i++) {
    	if(!isAlive[i])
    		continue;

    	for (int j=i+1; j<models.size(); j++) {
    		if (!isAlive[j])
    			continue;
    		if (cv::norm(positions[i] - positions[j]) < 60) {
    			isAlive[MIN(totalMatches[i], totalMatches[j]) == totalMatches[i]? i : j] = false;
    		}
    	}
    }

	///produce output
	for (int i=0; i< models.size(); i++) {
    	if (isAlive[i])
    		res[models[i]] = matches[i];
    }

    return res;
}



#endif //PROJECTWORK_MATCHING_H
