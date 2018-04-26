//
// Created by Marco DiVi on 25/04/18.
//

#pragma once


#include "RichImage.h"


cv::Mat _tmplMatch(cv::Mat &img, cv::Mat &model)
{
    cv::Mat result;

    matchTemplate( img, model, result, CV_TM_CCORR_NORMED );
    normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    return result;
}


///------- Localizing the best match with minMaxLoc ------------------------------------------------------------------------

cv::Point minmax( cv::Mat &result )
{
    double minVal, maxVal;
    cv::Point  minLoc, maxLoc, matchLoc;

    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
    matchLoc = minLoc;

    return matchLoc;
}


std::map<RichImage*, cv::Mat> templateMacth (std::vector<RichImage*> models, RichImage* scene) {

    std::map<RichImage*, cv::Mat> res;

    for (auto model : models){
         cv::Mat mat = _tmplMatch(scene->image, model->image);
         res[model] = mat;
    }

    return res;

}