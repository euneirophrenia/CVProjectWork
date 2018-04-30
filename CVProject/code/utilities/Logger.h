//
// Created by Marco DiVi on 26/04/18.
//

#pragma once

#include <string>
#include <iostream>

enum LogLevel {
    INFO = 0,
    WARNING = 1,
    DEBUG = 2
};

class Logger {

    public:
        static int logLevel;
        static inline void log(std::string msg, std::string tag = "[DEBUG]\t", LogLevel level = DEBUG, std::ostream& os = std::cerr) {
        	if (level <= logLevel)
            	os << tag << msg << "\n";
        }
};


int Logger::logLevel = INFO;