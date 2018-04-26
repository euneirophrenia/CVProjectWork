//
// Created by Marco DiVi on 26/04/18.
//

#pragma once

#include <string>
#include <iostream>

class Logger {

    public:
    static inline void log(std::string msg, std::string tag = "[DEBUG]\t", std::ostream& os = std::cerr) {
#ifdef DEBUG
        os << tag << msg << "\n";
#endif
    }

};