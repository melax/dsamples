# dclassify
cnn classification with depth and ir data

Yo, when you first clone from github, be sure to click the 'recursive' box if using tortoisegit so it will automatically grab librealsense into the third_party subfolder.

If you already cloned without getting the submodule, you can either try again, or just fetch the submodule after the fact by going to the git bash commandline and entering:
  * git submodule init
  * git submodule update

Note that the solution and project files are for visual studio 2015

A video showing how to use the app can be seen at:  
  https://www.youtube.com/watch?v=0umLZPGkbt4

You can save trained CNNs, either without (ctrl-s) or with (ctrl-t) the training data.
Look for generated .derp files exported in the same directory. 
Load a previously trained CNN on program startup by dragging trained_netXX.derp file onto the .exe 
or specify file as first parameter argv[1] when running from the commandline.

If you dont want the GUI, an additional project, load_trained_cnn, also shows how to just load the CNN and use it in code.  
With just ascii output, this latter project is only meant to be minimal reference code.  

See source code (its pretty short) for some additional information and instructions.

The program also can act as a http web server to allow it to work with a browser or even
the scratch offline editor: https://scratch.mit.edu/scratch2download/ 
There's a button in dclassify's gui to turn on the http server.
To use this in Scratch, Hold SHIFT while you left click to select the FILE drop down menu so the additional options show up.
Select "Import experimental HTTP extension", then browse to the dclassify subfolder and open the extension file: DClassify.s2e
The extension will show up in the "More Blocks" section.  

## FAQ


> How do I know when the training is completed?

You can usually see the error rate go down.  
Note that the program doesn't separate a test test from a training set, so its only the MSE error (average) on the most recent iterations.
You can just let it continue to train while you are using it to test your results by classifying the live input from camera.  
If the CNN error rate is low and not gettting any better, then you can probably shut off the training.


> What's the scoop on the slider widget for training?

That sets the number of backprop (training) iterations to do per frame.   this number increases exponentially. 
So increase this, but not too much - no reason to push it beyond the point that it slows down your framerate.   
Be sure to compile/run in release mode.  Debug mode for this program is quite slow for actual usage.


> Can I add samples during training or should I stop the training, add samples and restart the training?

No need to stop the training, just add samples as necessary - especially if you see the CNN output is not what it should be.  
Just mouseclick (or corresponding keypress) on the correct category while the live feed is on the object/thing you want to be classified.  


> My trained CNN continues to output one of my categories/labels even when there's none of my objects in front of the camera.

Add a category for 'everything else'.  pick a free category slot and collect samples while pointing the 
camera at background and other random non-objects that you want to rule out.


> What about RGB?

The CNN only uses ir and depth inputs.  
One thing that is nice about this is that both the depth and the IR lighting are fairly consistent.
This means that a trained net should continue to work as the ambient lighting in your environment changes (time of day, lights turned on/off etc).
So its not necessary to collect input samples under every possible lighting condition.  
That said, color could be easily added - just more input channels to the first convolution layer.


> When/why would I change the size of the CNN?

This can only be changed not only before any training has happened, but also before any samples are collected.  
The default smallest CNN only uses 16x16 input pixels, subsampling from the input area as necessary.
This makes for fast training which may be sufficient in many cases.  
In some situations, you may really need to have a bigger net and use more (perhaps 32x32 or more) input image.  
Note that this will typically require more input samples to train properly.


> How best to set the samping area and depth range?

In case anyone missed it, you can change the size of the cropped subwindow that is used as input to the CNN.  
Use the left mouse in either the depth or ir view of the data to change this value.
Note this doesn't change the size of the input to the CNN.  
This subwindow will be subsampled corresponding to the CNN input size.
Pick a reasonable size that fits around the objects you want to classify.
Also set a depth range that extends as much as needed and crops things beyond a distance you dont care about.


> What about changing these values after samples have been collected?

The CNN doesn't know the scale, absolute depth, or subsampling rate.   
It essentially is getting the grey pixels shown in the IR and Depth subsample subwindows.  
You'll have to provide enough samples to ensure coverage for all the situations that you want to train for.  
 

> Importance of data preprocessing?

Not really.  The header file for the depth camera does currently do a tiny bit of preprocessing on the depth data.  
It was just a small pass to remove "flying pixels".  Probably doesn't have a big impact on this app.
Probably not necessary since we see time and time again that machine learning systems 
deal with noisy data better than well intentioned (but actually information removing) cv code.


