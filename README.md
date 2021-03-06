
## DSamples
Samples that use depth camera data.

|  Sample       |  Description   |
| ------------- | -------------- |
|  dpca         |  point cloud analysis methods that provide useful and fairly consistent input for a number of use cases    |
|  dclassify    |  simple CNN based machine learning app with on-the-fly data collection, labelling, and training.           |
|  dphyspush    |  shows direct manipulation of physically simulated rigidbodies using point cloud data                      |

#### Release Info
| Software Version  |  Actual State  |  
| ----------------- | ------------- | 
| sqrt(-0.01)       |   code should run and be useful, but this is just an "imaginary unofficial test release" of code samples that are in the process of being made available in an alpha state.  | 

In other words, put here on github for early testing just to  make sure project files, submodules, and all source files are included and configured correctly. 
The point cloud or principal component analysis is a librealsense reimplementation of one of the samples from a IDF2013 tutorial session about depth cameras.  
Feel free to submit any feedback or suggestions via github issues.

### Details
- samples use librealsense api/library to get depth and ir data from realsense camera.
- samples developed and tested on r200 (aka ds4) realsense camera.  f200 and sr300 should work, let me know if not.
- samples only tested under windows OS.  the opengl-window initialization wrapper is windows specific at the moment.
- requires visual studio 2015.   A 2013 solution file is provided but it isn't regularly tested after each code commit, and thus might require minor fixes. 

#### git submodules
This repo depends on librealsense and sandbox repos on github.  
So if you are using tortoise git, be sure the recursive checkbox is checked to automatically fetch these dependencies.
Alternatively, if you prefer to use the command line, then open a git bash shell and enter commands:
```sh 
$ git submodule init 
$ git submodule update
```
With the repo and submodules cloned, open up the solution ( dsamples_vs2015.sln ) and everything should sucessfully build and run (assuming a depth camera is plugged in) on the first try.


