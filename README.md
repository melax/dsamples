
## DSamples
Samples that use depth camera data.

| ------------- | -------------- |
|  Sample       |  Description   |
| ------------- | -------------- |
|  dpca         |  point cloud analysis methods that provide useful and fairly consistent input for a number of use cases    |
|  dclassify    |  simple CNN based machine learning app with on-the-fly data collection, labelling, and training.           |
|  dphyspush    |  shows direct manipulation of physically simulated rigidbodies using point cloud data                      |
| ------------  | -------------- |

#### Release Info
| ----------------- | ------------- | ------ |
| software version  |  sqrt(-0.01)  |  this is just an "imaginary unofficial test release" of code samples that are in the process of being made available in an alpha state.  |
| ----------------- | ------------- | ------ |
in other words, put here on github for early testing just to  make sure project files, submodules, and all source files are included and configured correctly.
Feel free to submit any feedback or suggestions via github issues.

### Details
- samples use librealsense api/library to get depth and ir data from realsense camera.
- samples developed and tested on r200 (aka ds4) realsense camera.  f200 and sr300 should work, let me know if not.
- samples only tested under windows OS.  the opengl-window initialization wrapper is windows specific at the moment.
- requires visual studio 2015


This repo depends on librealsense and sandbox repos on github.  
So if you are using tortoise git, be sure the recursive checkbox is checked to automatically fetch these dependencies.
Then open up the solution and everything should sucessfully build and run (assuming a depth camera is plugged in) on the first try.

