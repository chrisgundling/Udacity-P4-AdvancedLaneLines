# Udacity-P4-AdvancedLaneLines
Computer Vision for Road Lane Finding using Vehicle Front Camera

By: Chris Gundling

---

The goals of this project were the following:
- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Code Summary
I used two different python notebooks for this project, the first (`P4-AdvancedLanes-Tuning.ipynb`) for developing an initial solution using single images at a time and tuning the various parameters. The second (`P4_AdvancedLaned-Video.pynb`) was for testing on the project video and in this version I have simplified many of the steps in order to speed up the image processing pipeline. The following sections will refer to both of these notebooks.

### Step 1: Camera Calibration

*Rubric: Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.*

I split up the camera calibration into three steps, which are shown in the first 3 code cells of `P4-AdvancedLanes-Tuning.ipynb`:
- 1a. Find the corners, undistort and warp
- 1b. Example of Image Calibration
- 1c. Example of Image Warping

I start by preparing "object points", which will be the (`x, y, z`) coordinates of the chessboard corners in the world. I assume the chessboard is fixed on the (`x, y`) plane at z=0, such that the object points are the same for each calibration image. `objp` is just a replicated array of coordinates, and `objpoints` is appended with a copy of it every time I successfully detect all chessboard corners in a test image. This happens for 17 of the calibration images when I use 9X6 for the number of corners. `imgpoints` are appended with the (`x, y`) pixel position of each of the corners in the image plane with each successful chessboard detection.

The output `objpoints` and `imgpoints` are then used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. Distortion correction to the test image is then completed using the `cv2.undistort()` function. The following images show the before and after:

<img src="images/calibrate.png" width="1000">

### Step 2: Pipeline (Single Images)

##### 1. Distortion Correction

*Rubric: Has the distortion correction been correctly applied to each image?*

The first step in my image processing pipeline was to undistort the image. The code for this step is in the 2nd cell of the  P4_Tuning.pynb notebook and produces following result for one of the test images:

#### 2. Color Transforms, Gradients and Other Methods

*Rubric: Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.*

I used several different techniques to create my final binary_warped image. I actually found that applying the perspective transform first gave better results and made it easier to see what the polynomial fits were trying to fit to. This is described in following sections, but I wanted to quickly mention it since all the transformed images have already been warped.

I first used color space transforms on my images, both to grayscale and to HSV color space. I found that using a combination of both these techniques helped to see both white and yellow lines. After applying the transform, the pixel values are thresholded by some amount. The threshold values were determined from extensive testing using all of the provided test images. 

Once the color space transform and thresholding is applied, I then used gradient direction, gradient magnitude and gradient thresholding to further process the images. The code for gradient x, y gradient thresholding is shown in section 3a, gradient magnitude in 3b, and gradient direction in 3c. Each of these techniques is able to pick up different aspects of the line pixels in each image.

Once the colorspace and gradient techniques were done, I combined several of these techniques to create the final binary_warped image. Once again, this required significant tuning based on the test images and the final video. The final line of code to create the binary_warped image and an example of the binary warped image are shown below:

#### 3. Perspective Transform

*Rubric: Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.*

As previously mentioned, the perspective transform was performed directly after distortion correction. I found it much easier to tune the various color space and gradient transforms when viewing the binary image from the bird’ eye view.

The code for my perspective transform is includes a function called birds_eye, which appears in lines 1 through 8 in the file P4_tuning.pynb (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook). The birds_eye() function takes as inputs an image ( img ), as well as source ( src ) and destination ( dst ) points. I chose the hardcode the source and destination points in the following manner:
```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

#### 4. Lane Line Curve Fitting (Histograms and Sliding Windows)

*Rubric: Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?*

After creating the binary_warped image, the next step was to apply a histogram to the image to determine the peak locations of pixels in the image. An example of this technique is shown below. Once the histogram was found, a sliding window technique was used to isolate the lane line pixels in the image. This was a critical step in the code for determining the proper left and right lane lines. 
The sliding windows (rectangles) start with their center at the X locations were the peak number of pixels were found. The sliding windows are then extended 100 pixels in either X direction (margin) and look for a minimum of 60 pixels. If this condition is met, then the, otherwise the windows will return to center. Once the lane pixel indices and X,Y locations are determined, a 2nd order polynomial fit was used to fit lines to these pixels for both the left and right lane lines. 

This next step is where I spent the most time and got the biggest gain in results. I used several different techniques to check that the determined polynomial fits for each lane line were realistic. 

1. Limits to Polynomial Coefficients: I first created several made up (random) curves that were similar in curvature to what the lane lines would be. This allowed me to get a good grasp for the values that I should be expecting for the polynomial coefficients. A plot of this is shown below. Using this information I limited the values that the squared and linear terms could take. 
2. Confidence of left/right lanes: Based on the number of pixels found corresponding to each lane line I implemented a “confidence” metric. This metric interesting to watch during the video processing and could be used in further pre-processing techniques.
3. Left Lane Right Lane Distance Apart: The distance between the left and right lanes should not change drastically, it should constantly be around 3.7 meters or 800 pixels in my implementation. I applied a check for this and if the lane lines were too far apart or too close together I would discard the information from the lane line with the least “confidence”. The discarded lane line was then given the pixels from the more confident lane line shifted by 800 pixels in the appropriate direction.

#### 5. Visualizing the Lanes and Calculating Lane Information

*Rubric: Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.*

I then created several sections of code that drew the lane lines on the image, determined the radius of curvature of the lines and determined the car’s position relative to the center of the lanes. 

#### 6. Lines on the Road!

*Rubric: Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.*

I then tested the full implementation of all these techniques on the 6 test images that were provided and the results can be seen below. Once I had achieved reasonably results on these test images, then I simplified as much of the code as I could (P4_Video.pynb) and ran the video through it. 

![alt text][image6]

---

### Step 3: Pipeline (Video)

*Rubric:	Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).*

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

*Rubric: Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?*

This was an eye opening experience to realize how much tuning was required to the color space, gradient and polynomial techniques to reach a successful result.  While the resulting code worked well for the project video, I found that it did not work particularly well on the challenge videos. For vehicle manufacturers to implement similar approaches that generalize well to all lane situations (or situations with no lane lines) seems like an incredibly difficult task. I implemented a metric in this project that I think could be used for further improvements in performance. By using an optimization technique, the “confidence” metric for each of lane lines detection could be maximized. This would be done by having all of the tunable parameters as inputs to the optimizer and the confidence as the output. The model would then be tested over a range of thousands (or millions) of different images and the tunable parameters would be selected such as to maximize the total lane line prediction confidence for all of the images.   
