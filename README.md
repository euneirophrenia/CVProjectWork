# CVProject

This repository contains everything about the projectwork for the CV exam.
It is meant for personal purposes only.

***

## Requirements
OpenCV 3.x.x, C++ at least standard 14, CMake 3.8.x, but should work on older versions too.

## Deplyment
it is a CMake project. Just build it with cmake and then compile with make.
Please, have care that the executable produced is placed in the same folder as the `settings.json` and the folder `models`.

> For a tentative Visual Studio project, please refer to [this branch](https://github.com/euneirophrenia/CVProjectWork/tree/visual-studio-support)

## Running
it is designed as a CL (command line) utility. 
At least, pass it as first argument the scene to work on (path relative to the executable folder) and then some options:
* `-ght` to use the generalized Hough transform
* `-time` to output time measurement
* `-sift` to use normal SIFT rather then RootSIFT

> There is also a `settings.json` file with some settings to tinker with.


## Recovery after everything exploded and nothing worked
I don't know, actually. I tried my best to make it work, if it does not for some reason I'm sorry.


## License
Use at your own risk and will. If you actually make some money off of this, teach me your secrets. 
