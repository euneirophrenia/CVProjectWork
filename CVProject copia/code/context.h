//
// Created by Marco DiVi on 05/03/18.
//

/**
 * Meant to include all the global variables, context informations and such.
 */

#ifndef OPENCV_CONTEXT_H
#define OPENCV_CONTEXT_H

#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#include <functional>
#include "boost/filesystem.hpp"
#include "json.hpp"
#include "tools.h"

#define CONFIG_PATH "../CVProject/settings.json"

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


/**
 * A function that returns a lambda (filename) => boolean, that returns true only if the "extension" of the provided file name
 * matches one of the accepted ones.
 * @param accepted the accepted extensions vector
 * @return a lambda (string -> boolean)
 */
std::function<bool(std::string)> fileExtensionFilter(std::vector<std::string> accepted) {

    return [&](std::string filename) {
        auto extension = boost::filesystem::extension(filename);
        for (auto& acc : accepted) {
            if (acc == extension)
                return true;
        }
        return false;
    };

}

enum Algorithms {
    BRISK = 0,
    ORB,
    SIFT,
    SURF
};

//TODO: in a future make it accept a config file and load configuration from there
//TODO: also make the public fields readonly
//TODO: make algorithms parameters a context variable, also read from file
class Context {

    private:
        Context() {
            std::ifstream inputstream;
            std::string json = sanifyJSON(CONFIG_PATH);
            auto config = nlohmann::json::parse(json);

            BASE_PATH = config["BASE_PATH"];
            auto acc=config["ACCEPTED_EXTENSIONS"];
            for (auto x : acc) {
                accepted_extensions.push_back(x);
            }
            read_directory(config["MODELS_PATH"], MODELS, fileExtensionFilter(accepted_extensions));
            read_directory(config["SCENES_PATH"], SCENES, fileExtensionFilter(accepted_extensions));
            MIN_MATCHES = config["MIN_MATCHES"];
        }


    public:
        std::string BASE_PATH;
        std::vector<std::string> MODELS;
        std::vector<std::string> SCENES;
        std::vector<std::string> accepted_extensions;
        int MIN_MATCHES;

        static Context& getInstance() {
            static Context instance;

            return instance;
        }

        Context(Context const&) = delete;
        void operator=(Context const&)  = delete;

};




#endif //OPENCV_CONTEXT_H
