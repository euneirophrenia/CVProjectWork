//
// Created by Marco DiVi on 07/04/18.
//

#ifndef PROJECTWORK_VOTINGMATRIX_H
#define PROJECTWORK_VOTINGMATRIX_H

#include "RichImage.hpp"
#include "blob.hpp"
#include "matching.hpp"


struct VotingMatrix {
    private:
    std::map<RichImage*, std::vector<Blob>> blobs;
    RichImage* scene;
    std::map<RichImage*, cv::Mat> masks;

    public:
    explicit VotingMatrix(RichImage* scene) {
        this->scene = scene;
    }

    inline void castVote(cv::DMatch match, RichImage *forModel, double collapseDistance = 5, bool useL1Norm = true) {
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
			std::stringstream ss;
			ss << "Vote @" << estimated_bary << " for " << forModel->path;
        	Logger::log(ss.str(), "[IGNORING]\t", DEBUG);
            return;
        }

        int found = masks[forModel].at<int>(b.position.y, b.position.x);
        if (found > 0) {
            blobs[forModel][found - 1] += b;
        }
        else {
            blobs[forModel].push_back(b);
            int n = (int)blobs[forModel].size();
            int i0 = (int) MAX(b.position.x - collapseDistance/2.0f, 0),
                    imax = (int) MIN(b.position.x + collapseDistance/2.0f, masks[forModel].cols ),
                    j0 = (int) MAX(b.position.y - collapseDistance/2.0f,0),
                    jmax = (int)MIN(b.position.y + collapseDistance/2.0f, masks[forModel].rows );

            for (int i = i0; i < imax; i++) {
                for (int j = j0; j < jmax; j++)
                    if (masks[forModel].at<int>(j, i) == 0)
                        masks[forModel].at<int>(j, i) = n;
            }
        }

    }

    inline void collapse(double collapsingDistance) {

        std::vector<Blob> collapsed;

        for (auto pair : blobs) {

            collapsed.clear();

            for (auto blob : pair.second) {

//                if (blob.position.x < 0 || blob.position.y < 0 ||
//                    blob.position.x > scene->image.cols  || blob.position.y > scene->image.rows) {
//                    Logger::log(blob, "[IGNORING]");
//                    continue;
//                }

                bool placed=false;
                for (int k = 0; k < collapsed.size() && !placed; k++){
                    double dist = distance(blob.position, collapsed[k].position);
                    if (dist <= collapsingDistance) {
                        placed = true;
                        collapsed[k] += blob;
                    }
                }

                if (!placed) collapsed.push_back(blob);

            }

            blobs[pair.first] = collapsed;

        }
    }

    inline void collapseConnected(double collapsingDistance) {

        for (auto pair : blobs) {
            cv::Mat votes = cv::Mat::zeros(scene->image.size(), CV_32F);
            for (auto blob : pair.second) {

                if (blob.position.x < collapsingDistance/2 || blob.position.y < collapsingDistance/2 ||
                    blob.position.x > votes.cols - collapsingDistance/2 || blob.position.y > votes.rows - collapsingDistance/2) {
                	if (context.cli_options->at("-debug"))
                    	Logger::log(blob,  "[IGNORING]\t");
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

            std::vector<Blob> compacted( howmany - 1);

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

    inline void prune(double pruneDistance) {
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
                        	std::stringstream ss;
                        	ss << allblobs[k] << " in favor of " << blob << "- " << dist;
                        	Logger::log(ss.str(), "[PRUNING]\t");
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
    inline void relativeFilter(double threshold = 0.5) {
        std::vector<size_t> indicesToRemove;
        for (auto pair:blobs) {

            double best=0;
            for (auto blob : pair.second) {
                if (blob.confidence > best)
                    best = blob.confidence;
            }
			if (context.cli_options->at("-debug")) {
				for (size_t i = 0; i < pair.second.size(); i++) {
					if (pair.second[i].confidence < threshold * best) {

						std::cerr << "[RELATIVE FILTERING] " << pair.second[i] << "\t("
								  << pair.second[i].confidence / best << "/"
								  << threshold << ")\n";
						indicesToRemove.push_back(i);
					}
				}

				blobs[pair.first] = erase_indices(blobs[pair.first], indicesToRemove);
				indicesToRemove.clear();
			}
			else {
            	auto todelete = std::remove_if(blobs[pair.first].begin(), blobs[pair.first].end(), [=](Blob b) {return b.confidence < threshold*best;} );
                blobs[pair.first].erase(todelete, blobs[pair.first].end());
			}
        }

    }

    ///filter out all blobs with less than an absolute value for confidence
    inline void absoluteFilter(double threshold = context["MIN_HOUGH_VOTES"]) {
        std::vector<size_t> indicesToRemove;

        for (auto pair:blobs) {
			if (context.cli_options->at("-debug")) {
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
			}
			else {
				auto todelete = std::remove_if(blobs[pair.first].begin(), blobs[pair.first].end(), [=](Blob b) {return b.confidence < threshold;} );
				blobs[pair.first].erase(todelete, blobs[pair.first].end());
			}
        }

    }

    inline std::vector<Blob> operator[] (RichImage* model) {
        return blobs[model];
    }

    inline std::map<RichImage*, std::vector<Blob>> asMap() {
        if (blobs.empty())
            return this->blobs;

        std::map<RichImage*, std::vector<Blob>> res;

        for (auto pair : blobs){
            if (!pair.second.empty())
                res[pair.first] = pair.second;
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

    inline void _ghtmatch(RichImage *model, Algorithm* algo, double spread = 10) {

        std::vector<cv::DMatch> matches = findKnn(scene->features, model->features, algo->matcher, context["THRESHOLD"], true);

        for (auto match : matches) {
                votes->castVote(match, model, spread);
        }

    }

    public:

    /// Leave default (negative) parameters to autotune them based on an estimate of the image scaling.
    /// (for better performance, provide proper values by hand)
    explicit GHTMatcher(RichImage* scene, double collapseDistance = -1, double pruneDistance = -1,
                        double relativeThreshold = 0.4, double absoluteThreshold = -1){
        votes = new VotingMatrix(scene);
        this->scene = scene;

        this->collapseDistance = collapseDistance >= 0 ? collapseDistance : scene->approximateScale() * 0.5;
        this->pruneDistance = pruneDistance >= 0 ? pruneDistance : scene->approximateScale() / 0.9;

        this->relativeThreshold = relativeThreshold;
        this->absoluteThreshold = absoluteThreshold;

    }

    std::map<RichImage*, std::vector<Blob>> GHTMatch(std::vector<RichImage *> models, Algorithm* algo) {

        if (scene->keypoints.empty())
            algo->detector->detectAndCompute(scene->image, cv::Mat(), scene->keypoints, scene->features);

        std::vector<Blob> allblobs;
        std::vector<size_t> indicesToRemove;
        for (auto mp : models) {
            _ghtmatch(mp, algo);
        }


		votes -> collapse(collapseDistance);
        votes -> relativeFilter(relativeThreshold);
        votes -> prune(pruneDistance);


        if (absoluteThreshold < 0) {
            float total_conf = 0;
            int total = 0;

            for (auto pair : votes->asMap()) {
                for (auto blob: pair.second) {
                    total += 1;
                    total_conf += blob.confidence;
                }
            }

            absoluteThreshold = 1 + 0.1 * total_conf/total;
        }

        ///filter out everything with confidence < 0.1 * mean_confidence, it is probably garbage
        votes->absoluteFilter(absoluteThreshold);

        return votes->asMap();
    }


    std::map<RichImage*, std::vector<Blob>> FastGHTMatch(std::vector<RichImage *> models, Algorithm* algo,
                                                        float** similarityMatrix = nullptr, float similarityTrheshold = 0.6) {

        if (scene->keypoints.empty())
            algo->detector->detectAndCompute(scene->image, cv::Mat(), scene->keypoints, scene->features);

        std::vector<cv::Mat> modelfeats;

        for (auto model : models) {
            if (model->features.empty())
                model->build(algo, true);
            modelfeats.push_back(model->features);
        }

        std::vector<cv::DMatch> dmatches;

        if (similarityMatrix == nullptr)
            dmatches = MultiFindKnn(modelfeats, scene->features, algo->matcher, context["THRESHOLD"]);
        else
            dmatches = MultiFindKnnWithSimilarity(modelfeats, scene->features, algo->matcher, similarityMatrix,
                                                  similarityTrheshold, context["THRESHOLD"]);

        for (auto match : dmatches) {
            this->votes->castVote(match, models[match.imgIdx], 10);
        }

        this->votes->collapse(collapseDistance);
        this->votes->relativeFilter(relativeThreshold);
        this->votes->prune(pruneDistance);

        if (absoluteThreshold < 0) {
            float total_conf = 0;
            int total = 0;

            for (auto pair : votes->asMap()) {
                for (auto blob: pair.second) {
                    total += 1;
                    total_conf += blob.confidence;
                }
            }

            absoluteThreshold = 1 + 0.1 * total_conf/total;
        }

        this->votes->absoluteFilter(absoluteThreshold);

        return this->votes->asMap();

    }

};

#endif //PROJECTWORK_VOTINGMATRIX_H