//
// Created by Marco DiVi on 26/04/18.
//

#pragma once

#include "RichImage.hpp"
#include <functional>
#include "matching.hpp"
#include "../utilities/Logger.hpp"

class Preprocesser {

    std::vector<RichImage*> _models;
    std::map<std::string, std::string> _ids;

    std::vector<RichImage*> easyToTell, hardToTell;

    float** similarity;

    public:
        explicit Preprocesser(int loadMode = cv::IMREAD_GRAYSCALE,
                std::function<std::string(std::string)> idMap = [](std::string s) { return fileName(s);}) {

            for (auto& p : context.MODELS) {
               _models.push_back(new RichImage(p, loadMode));
            }

            auto msize = context.MODELS.size();

            for (int i =0; i< msize; i++) {
                _ids[context.MODELS[i]] = idMap(context.MODELS[i]);
            }
        }

        ///To uniform the size of the models and blur if needed
        inline void uniform(int approximate_scale = -1) {

            cv::Size minimum;
            if (approximate_scale < 0) {
                minimum = _models[0]->image.size();

                for (auto image : _models) {
                    if (image->image.size().area() < minimum.area())
                        minimum = image->image.size();
                }
            }
            else {
                minimum.height = approximate_scale;
            }

            /// rescale each image so that its height matches the minimum one
            // I chose height arbitrarily, I could have chosen the width, just not both, i want to preserve proportions
            for (auto image : _models) {
                float factor = 1.5f * minimum.height / image->image.size().height  ;
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

        inline void build(Algorithm* alg) {
            for (auto m : _models){
            	//m->deBlur(false);
                m->build(alg, true);  //we also want the hoguh model, hence the "true"
            }
        }

        inline void computeSimilarity(Algorithm* alg, float threshold = 0.1) {
            similarity = new float*[_models.size()];

            Logger::log("Computing similarity table...", "[INFO]\t", INFO);

			std::vector<bool> placed(_models.size());

            for (int i=0; i<_models.size(); i++) {
                similarity[i] = new float[_models.size()];
            }


            for (int i=0; i<_models.size(); i++) {
                similarity[i][i] = 1.0f;
                if (_models[i]->isHard)
                    continue;

                for (int j=i+1; j < _models.size(); j++) {
                    if (_models[j]->isHard)
                        continue;

                    auto matches = findKnn(_models[i]->features, _models[j]->features, alg->matcher);
                    similarity[i][j] = 1.0f* matches.size() / _models[i]->features.rows;
                    similarity[j][i] = 1.0f* matches.size() / _models[j] -> features.rows;

                    Logger::log(_models[i]->path + "\t" +_models[j]->path + "\t" + std::to_string(similarity[i][j]), "[SIMILARITY]\t");

                    if (similarity[i][j] > threshold || similarity[j][i] > threshold) {
                        if (!_models[i]->isHard)
                            _models[i]->isHard=true;
                        if (!_models[j]->isHard)
                            _models[j]->isHard=true;
                    }
                }

            }

            for (int i=0; i< _models.size(); i++){
                if (_models[i]->isHard)
                    hardToTell.push_back(_models[i]);
                else
                    easyToTell.push_back(_models[i]);
            }

			Logger::log("..Done", "[INFO]\t", INFO);
        }

        inline std::string idOf(std::string model_name) {
            return _ids[model_name];
        }

        inline std::vector<RichImage*> models(){
            return _models;
        }

        inline std::vector<RichImage*> easyModels() {
            return easyToTell;
        }

        inline std::vector<RichImage*> hardModels() {
            return hardToTell;
        }

        inline float** similarityTable(){
        	return similarity;
        }


};