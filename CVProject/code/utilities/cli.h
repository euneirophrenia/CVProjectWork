//
// Created by Marco DiVi on 29/04/18.
//

#pragma once

///A utility to parse command line arguments without littering too much the main

#include <string>
#include <cstring>
#include <vector>
#include <map>
#include <iostream>

std::vector<std::string> accepted_options = {
		"-debug",    ///enable debug prints (warning: due to some memory trick i did to ease debug, it may lead to data races)
		"-ght",     /// enable ght matching
		"-time",   /// log times
		"-sift",  /// use normal sift rather than rootsift
};


void detect(int argc, char** argv, std::string* scene) {

	if (argc < 2) {
		std::cerr << "At least I need the test scene to work with (relative path from project folder).\n";
		std::cerr << "Recognized options are the following:\n";
		std::cerr << "\t-debug   to enable debug outputs (warning: may lead to data races);\n";
		std::cerr << "\t-ght     enables the generalized hough transform;\n";
		std::cerr << "\t-time    enables time performance logging;\n";
		std::cerr << "\t-sift    uses normal SIFT rather then RootSIFT;\n";
		exit(1);

	}
	*scene = argv[1];

	for (int i=0; i< accepted_options.size(); i++)
		(*context.cli_options)[accepted_options[i]] =  false;

	for (int i=2; i<argc; i++) {
		if ((*context.cli_options).count(argv[i]) == 0){
			std::cerr << "Unrecognized option:\t" << argv[i] << "\n";
			continue;
		}
		context.cli_options->at(argv[i]) = true;
	}

}
