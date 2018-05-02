# CVProject

This repository contains everything about the projectwork for the CV exam.
It is meant for personal purposes only.

***

## Requirements
OpenCV 3.x.x, C++ at least standard 14, CMake 3.8.x, but should work on older versions too.

## Deplyment
it is a CMake project (no Visual Studio project *yet*). Just build it with cmake and then compile with make.

## Running
it is designed as a CL (command line) utility. At least, pass it as first argument the scene to work on (path relative to project folder, i.e. scenes/e1.png) and then some options:
* `-ght` to use the generalized Hough transform
* `-time` to output time measurement
* `-sift` to use normal SIFT rather then RootSIFT

> There is also a `settings.json` file with some settings to tinker with.


## Recovery after everything exploded and nothing worked
I don't know, actually. I tried my best to make it work, if it does not for some reason I'm sorry.


## License
Use at your own risk and will. If you actually make some money off of this, teach me your secrets. 
