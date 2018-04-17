//
// Created by Marco DiVi on 07/04/18.
//

#ifndef PROJECTWORK_VOTINGMATRIX_H
#define PROJECTWORK_VOTINGMATRIX_H

#include "RichImage.h"
#include "blob.h"
#include "matching.h"


struct VotingMatrix {
    private:
    std::map<RichImage*, std::vector<Blob>> blobs;
    RichImage* scene;
    std::map<RichImage*, cv::Mat> masks;

    public:
    explicit VotingMatrix(RichImage* scene) {
        this->scene = scene;
    }

    void castVote(cv::DMatch match, RichImage *forModel, double collapseDistance = 5, bool useL1Norm = true) {
        double scale = scene->keypoints[match.trainIdx].size;
        double angle = scene->keypoints[match.trainIdx].angle - forModel->keypoints[match.queryIdx].angle;

        cv::Point2d scenept = scene->keypoints[match.trainIdx].pt;
        cv::Point2d estimated_bary;
        cv::Vec2d houghmodel = forModel->houghModel[match.queryIdx];


        houghmodel = rotate(houghmodel, angle);

        estimated_bary.x = (scenept.x +  scale * houghmodel[0]);
        estimated_bary.y = (scenept.y +  scale * houghmodel[1]);

        if (masks.count(forModel) == 0)
            masks[forModel] = cv::Mat::zeros(scene->image.size(), CV_32S);

        Blob b;
        b.position = estimated_bary;
        b.confidence = 1;
        b.matches.push_back(match);
        b.area = 1;
        b.model = forModel;

        if (!b.isInside(scene->image)) {
#ifdef DEBUG
            std::cerr << "[IGNORING] Vote @" << estimated_bary << " for " << forModel->path <<"\n";
#endif
            return;
        }

        int found = masks[forModel].at<int>(b.position);
        if (found > 0) {
            blobs[forModel][found - 1] += b;
        }
        else {
            blobs[forModel].push_back(b);
            for (int i = b.position.x - collapseDistance/2.0f; i < b.position.x + collapseDistance/2.0f; i++) {
                for (int j = b.position.y - collapseDistance/2.0f; j < b.position.y + collapseDistance/2.0f; j++)
                    masks[forModel].at<int>(j, i) = (int)blobs[forModel].size();
            }
        }

    }

    void collapse(double collapsingDistance) {

        std::vector<Blob> collapsed;

        for (auto pair : blobs) {

            collapsed.clear();

            for (auto blob : pair.second) {

                if (blob.position.x < 0 || blob.position.y < 0 ||
                    blob.position.x > scene->image.cols  || blob.position.y > scene->image.rows) {
#ifdef DEBUG
                    std::cerr << "[IGNORING] " << blob << "\n";
#endif
                    continue;
                }

                bool placed=false;
                for (int k = 0; k < collapsed.size() && !placed; k++){
                    double dist = cv::norm(blob.position - collapsed[k].position);
                    if (dist <= collapsingDistance) {
#ifdef DEBUG
                        std::cerr << "[COLLAPSING] " << blob <<  " with " << collapsed[k] << " - " << dist << "\n";
#endif
                        placed = true;
                        collapsed[k] += blob;
                    }
                }

                if (!placed) collapsed.push_back(blob);

            }

            blobs[pair.first] = collapsed;

        }
    }

    void collapseConnected(double collapsingDistance) {

        for (auto pair : blobs) {
            cv::Mat votes = cv::Mat::zeros(scene->image.size(), CV_32F);
            for (auto blob : pair.second) {

                if (blob.position.x < collapsingDistance/2 || blob.position.y < collapsingDistance/2 ||
                    blob.position.x > votes.cols - collapsingDistance/2 || blob.position.y > votes.rows - collapsingDistance/2) {
#ifdef DEBUG
                    std::cerr << "[IGNORING] " << blob << "\n";
#endif
                    continue;
                }

                for (int x = int(blob.position.x - collapsingDistance/2) ; x < blob.position.x + collapsingDistance/2 ; x++){
                    for (int y = int(blob.position.y - collapsingDistance/2) ; y < blob.position.y + collapsingDistance/2 ; y++){
                        double dist = cv::norm(blob.position - cv::Point2d(y,x));
                        votes.at<float>(y,x) += 1;
                    }
                }
            }
            cv::Mat labels(scene->image.size(), CV_32S);
            int howmany = cv::connectedComponents(votes > 0, labels);
            if (howmany < 2) {
                continue;
            }

            std::vector<Blob> compacted;
            compacted.reserve( howmany - 1);
            for (int w = 0; w < howmany-1; w++) {
                Blob prototype;
                prototype.model =  pair.first;
                prototype.confidence = 0;
                prototype.area = 0;
                prototype.position=cv::Point2d(0,0);
                prototype.matches = std::vector<cv::DMatch>();
                compacted.push_back(prototype);
            }

            for (auto b : pair.second) {
                if (scene->contains(b.position)) {
                    int label = labels.at<int>(b.position);
                    if (label == 0) {
                        continue;
                    }
                    compacted[label - 1] += b;
                }
            }

            blobs[pair.first] = compacted;

        }
    }

    void prune(double pruneDistance) {
        std::vector<Blob> allblobs;
        std::vector<size_t> indicesToRemove;

        for (auto pair : blobs) {
            auto matches = pair.second;

            for (auto blob : matches) {
                indicesToRemove.clear();
                bool best=true;

                for (int k =0; k<allblobs.size() && best; k++){
                    double dist = distance(allblobs[k].position, blob.position);
                    if (allblobs[k].confidence >= blob.confidence &&  dist <= pruneDistance) {
                        best = false;
                    }

                    if (allblobs[k].confidence < blob.confidence && dist <= pruneDistance) {
                        indicesToRemove.push_back(k);
                        if (blob.model == allblobs[k].model) {
                            continue;
                        }
                        else {
#ifdef DEBUG
                            std::cerr << "[PRUNING] " << allblobs[k].model ->path << " (" << allblobs[k].position << ", " <<
                                      allblobs[k].confidence << ") in favor of " << blob.model->path << " (" << blob.position
                                      << ", " <<
                                      blob.confidence << " ) - " << dist << "\n";
#endif
                        }
                    }
                }

                if (best)
                    allblobs.push_back(blob);

                allblobs = erase_indices(allblobs, indicesToRemove);
            }

        }

        blobs.clear();
        for (auto b : allblobs) {
            blobs[b.model].push_back(b);
        }
    }

    ///filter out all blobs with confidence < threhsold * best_confidence within the same model
    void relativeFilter(double threshold = 0.5) {
        std::vector<size_t> indicesToRemove;
        for (auto pair:blobs) {

            double best=0;
            for (auto blob : pair.second) {
                if (blob.confidence > best)
                    best = blob.confidence;
            }
#ifdef DEBUG
            for (size_t i=0; i < pair.second.size(); i++) {
                if (pair.second[i].confidence < threshold * best) {

                    std::cerr << "[RELATIVE FILTERING] " << pair.second[i] << "\t(" << pair.second[i].confidence/best << "/"
                              << threshold << ")\n";
                    indicesToRemove.push_back(i);
                }
            }

            blobs[pair.first] = erase_indices(blobs[pair.first], indicesToRemove);
            indicesToRemove.clear();
#else
            auto todelete = std::remove_if(blobs[pair.first].begin(), blobs[pair.first].end(), [=](Blob b) {return b.confidence < threshold*best;} );
                blobs[pair.first].erase(todelete, blobs[pair.first].end());
#endif
        }

    }

    ///filter out all blobs with less than an absolute value for confidence
    void absoluteFilter(double threshold = context["MIN_HOUGH_VOTES"]) {
        std::vector<size_t> indicesToRemove;

        for (auto pair:blobs) {
#ifdef DEBUG
            for (size_t i = 0; i < pair.second.size(); i++) {

                if (pair.second[i].confidence < threshold) {
                    indicesToRemove.push_back(i);
                    std::cerr << "[ABSOLUTE FILTERING] " << pair.second[i] << "\t(" << pair.second[i].confidence
                              << "/"
                              << threshold << ")\n";
                }

                blobs[pair.first] = erase_indices(blobs[pair.first], indicesToRemove);
                indicesToRemove.clear();
            }
#else
            auto todelete = std::remove_if(blobs[pair.first].begin(), blobs[pair.first].end(), [=](Blob b) {return b.confidence < threshold;} );
                blobs[pair.first].erase(todelete, blobs[pair.first].end());
#endif
        }

    }

    std::vector<Blob> operator[] (RichImage* model) {
        return blobs[model];
    }

    std::map<RichImage*, std::vector<Blob>> asMap() {
        return this->blobs;
    }

};


struct SimplisticVotingMatrix {

    private:
    static std::vector<Blob> _ghtmatch(RichImage *model, RichImage *scene, Algorithm algo) {

        cv::Mat votes = cv::Mat::zeros(scene->image.rows, scene->image.cols, CV_32FC1);

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

            if (estimated_bary.x < scale/2 || estimated_bary.y < scale/2 ||
                estimated_bary.x > votes.cols - scale/2  || estimated_bary.y > votes.rows - scale/2) {
                std::cerr << "Ignoring " << estimated_bary << "\n";
                //todo:: make it ~rubberband on the border, it may still provide useful insights
                continue;
            }

            for (int x = int(estimated_bary.x - scale/2); x<= int(estimated_bary.x + scale/2); x++) {
                for (int y = int(estimated_bary.y - scale/2); y<= int(estimated_bary.y + scale/2); y++) {
                    auto dist = (estimated_bary.x - x)*(estimated_bary.x - x) + (estimated_bary.y - y)*(estimated_bary.y - y);
                    votes.at<float>(y,x) += exp(-dist/8.0) ;
                }
            }

        }

        return aggregate(votes, model);
    }

    public:

    static std::map<std::string, std::vector<Blob>> GHTMatch(std::vector<RichImage *> models, RichImage *scene, Algorithm algo) {

        std::map<std::string, std::vector<Blob>> res;

        if (scene->keypoints.empty())
            algo.detector->detectAndCompute(scene->image, cv::Mat(), scene->keypoints, scene->features);
        for (auto model : models) {
            if (model->features.empty())
                model->build(&algo, true);
        }


        std::vector<Blob> allblobs;
        std::vector<size_t> indicesToRemove;
        for (int i=0; i<models.size(); i++){
            res[models[i]->path] = std::vector<Blob>();

            auto matches = _ghtmatch(models[i], scene, algo);

            for (auto blob : matches) {
                indicesToRemove.clear();
                bool best=true;

                for (int k =0; k<allblobs.size() && best; k++){
                    double dist = distance(allblobs[k].position, blob.position);
                    if (allblobs[k].confidence >= blob.confidence &&  dist <= scene->approximateScale()/4) {
                        best = false;
                    }

                    if (allblobs[k].confidence < blob.confidence && dist <= scene->approximateScale()/4) {
                        indicesToRemove.push_back(k);
                        if (blob.model == allblobs[k].model) {
                            std::cerr << "[ASSIMILATING] " << allblobs[k].model->path << " (" << allblobs[k].position << ", " <<
                                      allblobs[k].confidence << ") with " << blob.model->path << " (" << blob.position << ", " <<
                                      blob.confidence << " ) - " << dist << "\n";
                            blob += allblobs[k];
                        }
                        else {
                            std::cerr << "[PRUNING] " << allblobs[k].model->path << " (" << allblobs[k].position << ", " <<
                                      allblobs[k].confidence << ") in favor of " << blob.model->path << " (" << blob.position
                                      << ", " <<
                                      blob.confidence << " ) - " << dist << "\n";
                        }
                    }
                }

                if (best)
                    allblobs.push_back(blob);

                allblobs = erase_indices(allblobs, indicesToRemove);
            }

        }

        for (auto blob: allblobs) {
            res[blob.model->path].push_back(blob);
        }


        return res;
    }

    static std::map<std::string, std::vector<Blob>> FastGHTMatch(std::vector<RichImage *> models, RichImage *scene, Algorithm algo) {
        std::map<std::string, std::vector<Blob>> res;

        if (scene->keypoints.empty())
            algo.detector->detectAndCompute(scene->image, cv::Mat(), scene->keypoints, scene->features);

        std::vector<cv::Mat> modelfeats;

        for (auto model : models) {
            if (model->features.empty())
                model->build(&algo, true);
            modelfeats.push_back(model->features);
        }

        auto dmatches = MultiFindKnn(modelfeats, scene->features, algo.matcher, context["THRESHOLD"]);

        std::vector<cv::Mat> votes;
        std::vector<cv::Mat> scales;

        for (int i=0; i < models.size(); i++) {
            cv::Mat z1 = cv::Mat::zeros(scene->image.rows, scene->image.cols, CV_32FC1);
            cv::Mat z2 = cv::Mat::zeros(scene->image.rows, scene->image.cols, CV_32FC1);
            votes.push_back(z1);
            scales.push_back(z2);
        }

        for (auto match : dmatches) {
            double scale = scene->keypoints[match.queryIdx].size;
            double angle = scene->keypoints[match.queryIdx].angle - models[match.imgIdx]->keypoints[match.trainIdx].angle;

            cv::Point2d scenept = scene->keypoints[match.queryIdx].pt;
            cv::Point2d estimated_bary;
            cv::Vec2d houghmodel = models[match.imgIdx]->houghModel[match.trainIdx];


            houghmodel = rotate(houghmodel, angle);

            estimated_bary.x = (scenept.x +  scale * houghmodel[0]);
            estimated_bary.y = (scenept.y +  scale * houghmodel[1]);

            if (estimated_bary.x < scale/2 || estimated_bary.y < scale/2 ||
                estimated_bary.x > votes[match.imgIdx].cols - scale/2 || estimated_bary.y > votes[match.imgIdx].rows - scale/2)
            {
                std::cerr << "Ignoring " << estimated_bary << "\n";
                //todo:: make it ~rubberband on the border, it may still provide useful insights
                continue;
            }

            for (int x = int(estimated_bary.x - scale/2); x<= int(estimated_bary.x + scale/2); x++) {
                for (int y = int(estimated_bary.y - scale/2); y<= int(estimated_bary.y + scale/2); y++) {
                    auto dist = (estimated_bary.x - x)*(estimated_bary.x - x) + (estimated_bary.y - y)*(estimated_bary.y - y);
                    votes[match.imgIdx].at<float>(y,x) += exp(-dist/8.0) ;
                }
            }

        }

        std::vector<Blob> allblobs;
        std::vector<size_t> indicesToRemove;

        for (int i=0; i< models.size(); i++) {
            auto matches = aggregate(votes[i], models[i]);
            res[models[i]->path] = std::vector<Blob>();

            for (auto blob : matches) {
                indicesToRemove.clear();
                bool best=true;

                for (int k =0; k<allblobs.size() && best; k++){
                    double dist = distance(allblobs[k].position, blob.position);
                    if (allblobs[k].confidence >= blob.confidence &&  dist <= scene->approximateScale()/4) {
                        best = false;
                    }

                    if (allblobs[k].confidence < blob.confidence && dist <= scene->approximateScale()/4) {
                        indicesToRemove.push_back(k);
                        if (blob.model  == allblobs[k].model) {
                            std::cerr << "[ASSIMILATING] " << allblobs[k].model->path << " (" << allblobs[k].position << ", " <<
                                      allblobs[k].confidence << ") with " << blob.model->path << " (" << blob.position << ", " <<
                                      blob.confidence << " ) - " << dist << "\n";
                            blob += allblobs[k];
                        }
                        else {
                            std::cerr << "[PRUNING] " << allblobs[k].model->path << " (" << allblobs[k].position << ", " <<
                                      allblobs[k].confidence << ") in favor of " << blob.model->path << " (" << blob.position
                                      << ", " <<
                                      blob.confidence << " ) - " << dist << "\n";
                        }
                    }
                }

                if (best)
                    allblobs.push_back(blob);

                allblobs = erase_indices(allblobs, indicesToRemove);
            }

        }

        for (auto blob: allblobs) {
            res[blob.model->path].push_back(blob);
        }


        return res;

    }

};

///A class to handle the ght matching
class GHTMatcher {

    private:
    RichImage* scene;
    double collapseDistance;
    double pruneDistance;
    double relativeThreshold;
    double absoluteThreshold;

    VotingMatrix* votes;

    void _ghtmatch(RichImage *model, Algorithm* algo) {

        if (model->keypoints.empty()) {
            model->build(algo, true);
        }


        std::vector<cv::DMatch> matches = findKnn(scene->features, model->features, algo->matcher, context["THRESHOLD"], true);

        for (auto match : matches) {
            votes->castVote(match, model, 10);
        }

    }

    public:

    /// Leave default (negative) parameters to autotune them based on an estimate of the image scaling.
    /// (for better performance, provide proper values by hand)
    explicit GHTMatcher(RichImage* scene, double collapseDistance = -1, double pruneDistance = -1,
                        double relativeThreshold = 0.4, double absoluteThreshold = -1){
        votes = new VotingMatrix(scene);
        this->scene = scene;

        this->collapseDistance = collapseDistance >= 0 ? collapseDistance : scene->approximateScale()/8;
        this->pruneDistance = pruneDistance >= 0 ? pruneDistance : scene->approximateScale()/1.5;

        this->relativeThreshold = relativeThreshold;
        this->absoluteThreshold = absoluteThreshold;

    }

    std::map<RichImage*, std::vector<Blob>> GHTMatch(std::vector<RichImage *> models, Algorithm* algo) {

        if (scene->keypoints.empty())
            algo->detector->detectAndCompute(scene->image, cv::Mat(), scene->keypoints, scene->features);

        for (auto model : models) {
            if (model->features.empty())
                model->build(algo, true);
        }


        std::vector<Blob> allblobs;
        std::vector<size_t> indicesToRemove;
        for (auto mp : models) {
            _ghtmatch(mp, algo);
        }

        votes -> collapse(collapseDistance);

        if (absoluteThreshold < 0) {
            float total_conf = 0;
            int total = 0;

            for (auto pair : votes->asMap()) {
                for (auto blob: pair.second) {
                    total += 1;
                    total_conf += blob.confidence;
                }
            }

            absoluteThreshold = 0.1 * total_conf/total;
        }

        ///filter out everything with confidence < 0.1 * mean_confidence, it is probably garbage
        votes->absoluteFilter(absoluteThreshold);
        votes->relativeFilter(relativeThreshold);

        votes->prune(pruneDistance);

        return votes->asMap();
    }

    std::map<RichImage*, std::vector<Blob>> FastGHTMatch(std::vector<RichImage *> models, Algorithm* algo) {

        if (scene->keypoints.empty())
            algo->detector->detectAndCompute(scene->image, cv::Mat(), scene->keypoints, scene->features);

        std::vector<cv::Mat> modelfeats;

        for (auto model : models) {
            if (model->features.empty())
                model->build(algo, true);
            modelfeats.push_back(model->features);
        }

        auto dmatches = MultiFindKnn(modelfeats, scene->features, algo->matcher, context["THRESHOLD"]);

        for (auto match : dmatches) {
            this->votes->castVote(match, models[match.imgIdx], collapseDistance);
        }

        //this->votes->collapse(collapseDistance);
        this->votes->relativeFilter(relativeThreshold);
        this->votes->prune(pruneDistance);

        return this->votes->asMap();

    }


};

#endif //PROJECTWORK_VOTINGMATRIX_H
