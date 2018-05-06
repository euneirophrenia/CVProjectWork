# CVProject

This repository contains everything about the projectwork for the CV exam.
It is meant for personal purposes only.

***

## Requirements
OpenCV 3.x.x, C++ at least standard 14, Windows 64 bit architecture.
Preferibly Visual Studio 17, but should work on elder versions too (after the automatic conversion).

## Deplyment
This is the Visual Studio-compatible version.

Compile it through Visual Studio, it should output an exe (ShelfDetect) in the base folder of the project. Run it from a terminal (powershell, ideally) with proper options (see below).
It needs to have the `opencv3\x64\vc14\bin` folder added to the path (it contains the various libraries, compiled with the non free modules). Everything else *should* be already properly set in the project settings.



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
