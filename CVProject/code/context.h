//
// Created by Marco DiVi on 05/03/18.
//

/**
 * Meant to include all the global variables, context informations and such.
 */

#ifndef OPENCV_CONTEXT_H
#define OPENCV_CONTEXT_H

#define DEBUG

#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#include <functional>

#include "utilities/json.hpp"
#include "utilities/tools.h"

#define CONFIG_PATH "../CVProject/settings.json"

struct Algorithm {
    cv::DescriptorMatcher* matcher;
    cv::FeatureDetector* detector;

    explicit Algorithm(cv::FeatureDetector* detector, cv::DescriptorMatcher* matcher){
        this->matcher = matcher;
        this->detector = detector;
    }
};

/**
 * Populates the provided vector with the names of the files in the provided directory matching the given criterium.
 * @param name the directory path
 * @param v the output vector
 * @param filterName a function accepting a string (name of a file) and returning a boolean (default function accepts every file)
 */
void read_directory(const std::string& name, std::vector<std::string> & v, std::function<bool(std::string)> filterName = [](std::string name) { return true;} )
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != nullptr) {
        if (filterName(dp->d_name))
            v.emplace_back(name + "/" + dp->d_name);
    }
    closedir(dirp);
}


std::string extension(std::string& filename) {
    auto dot = filename.find_last_of('.');
    if (dot == std::string::npos) {
        throw std::invalid_argument("No extension found");
    }

    return filename.substr(dot , filename.size());

}

/**
 * A function that returns a lambda (filename) => boolean, that returns true only if the "extension" of the provided file name
 * matches one of the accepted ones.
 * @param accepted the accepted extensions vector
 * @return a lambda (string -> boolean)
 */
std::function<bool(std::string)> fileExtensionFilter(std::vector<std::string> accepted) {

    return [&](std::string filename) {
        auto extension_ = extension(filename);
        for (auto& acc : accepted) {
            if (acc == extension_)
                return true;
        }
        return false;
    };

}

/*enum Algorithms {
    BRISK = 0,
    ORB,
    SIFT,
    SURF
};*/

//TODO: also make the public fields readonly
//TODO: make algorithms parameters a context variable, also GRAY_SCALE / COLOR
class Context {

    private:
        std::string _json = sanifyJSON(CONFIG_PATH);
        nlohmann::json raw_configuration = nlohmann::json::parse(_json);

        Context() {

            BASE_PATH = raw_configuration["BASE_PATH"];
            auto acc=raw_configuration["ACCEPTED_EXTENSIONS"];
            for (auto& x : acc) {
                accepted_extensions.push_back(x);
            }
            read_directory(raw_configuration["MODELS_PATH"], MODELS, fileExtensionFilter(accepted_extensions));
            read_directory(raw_configuration["SCENES_PATH"], SCENES, fileExtensionFilter(accepted_extensions));
            MIN_MATCHES = raw_configuration["MIN_MATCHES"];
            GAUSSIAN_KERNEL_SIZE = CvSize(raw_configuration["GAUSSIAN_KERNEL_SIZE"][0], raw_configuration["GAUSSIAN_KERNEL_SIZE"][1]);
            GAUSSIAN_X_SIGMA = raw_configuration["GAUSSIAN_X_SIGMA"];
            GAUSSIAN_Y_SIGMA = raw_configuration["GAUSSIAN_Y_SIGMA"];
            GOOD_MATCH_RATIO_THRESHOLD = raw_configuration["THRESHOLD"];
            KDTREES_INDEX = raw_configuration["KDTREES_INDEX"];
        }


    public:
        std::string BASE_PATH;
        std::vector<std::string> MODELS;
        std::vector<std::string> SCENES;
        std::vector<std::string> accepted_extensions;
        int MIN_MATCHES;
        cv::Size GAUSSIAN_KERNEL_SIZE;
        float GAUSSIAN_X_SIGMA, GAUSSIAN_Y_SIGMA;
        float GOOD_MATCH_RATIO_THRESHOLD;
        int KDTREES_INDEX;


        static Context& getInstance() {
            static Context instance;

            return instance;
        }

        Context(Context const&) = delete;
        void operator=(Context const&)  = delete;

        /// meant to be used only in extrema ratio, it's a bit more safer to use the public fields, instead of hardcoding keys everywhere
        //with this general scehma, the keys are hard coded only here, maybe in the future i will move towards some #define
        const nlohmann::json::value_type& operator[](const std::string& key) const {
            //std::cerr << "WARNING: accessing context informations by key is not something I'd advise right now. Be careful.\n";
            if (raw_configuration.count(key) == 0) {
                throw std::invalid_argument("Key not found " + key);
            }
            return raw_configuration[key];
        }

};


#endif //OPENCV_CONTEXT_H
