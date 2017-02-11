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
I used two different python notebooks for this project, the first (`P4-AdvancedLanes-Tuning.ipynb`) for developing an initial solution using single images at a time and tuning the various parameters. The second (`P4_AdvancedLaned-Video.pynb`) was for testing on the project video. In the video notebook I have simplified many of the steps in order to speed up the image processing pipeline. The following sections will refer mostly (`P4-AdvancedLanes-Tuning.ipynb`) and the video section will refer to (`P4_AdvancedLaned-Video.pynb`).

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

The first step in my image processing pipeline was to undistort the image using the camera calibration information. The code for this step is at `Step 2: Apply Calibration to Raw Images` and produces the following result for one of the test images:

<img src="images/real_calibrate.png" width="1000">

#### 2. Color Transforms, Gradients and Other Methods

*Rubric: Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.*

I used several different techniques to create my final binary_warped image. I actually found that applying the perspective transform first gave better results and made it easier to see what the polynomial fits were trying to fit to. This is described in following sections, but I wanted to quickly mention it since all the transformed images have already been warped.

In `Steps 3b-3e` in `P4-AdvancedLanes-Tuning.ipynb` I first used color space transforms on my images, both to grayscale and to HLS color space. I found that using a combination of both these techniques helped to see both white and yellow lines. After applying the transform, the pixel values are thresholded by (gray threshold: `thresh_g = (180, 255)` and HLS threshold: `thresh_s = (140, 255)`). The threshold values were determined from extensive testing using all of the provided test images. Only the saturation channel of the HLS colorspace was used as this proved particulary useful in seeing the yellow lines.

Once the color space transform and thresholding is applied, I then used functions that applied Sobel Threshholding in the x and y orientations, gradient magnitude and directional gradient thresholding to further process the images. Each of these techniques is able to pick up different aspects of the line pixels in each image. The following shows the final inputs to each of these functions.

```
# Sobel X on HLS
gradx_s = abs_sobel_thresh(bird, orient='x', thresh_min=5, thresh_max=100, HLS=True) 

# Sobel Y on HLS
grady_s = abs_sobel_thresh(bird, orient='y', thresh_min=5, thresh_max=100, HLS=True) 

# Magnitude on HLS
mag_binary_s = mag_thresh(bird, sobel_kernel=9, mag_thresh=(50, 200), HLS=True) 

# Magnitude on Grayscale
mag_binary = mag_thresh(bird, sobel_kernel=9, mag_thresh=(50, 200),HLS=False) 

# Directional on HLS
dir_binary_s = dir_threshold(bird, sobel_kernel=15, thresh=(0.7, 1.3), HLS=True) 
```

Once the colorspace and gradient techniques were performed, I combined several of these techniques to create the final `binary_warped` image. Once again, this required significant tuning based on the test images and the final video. The final line of code to create the `binary_warped` image and an example of the binary warped image are shown below:

`binary_warped[((gradx_s == 1) & (grady_s == 1)) | (mag_binary_s == 1) | (dir_binary_s == 1) | (mag_binary == 1)] = 1`

<img src="images/final_binary_1.png" width="1000">

#### 3. Perspective Transform

*Rubric: Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.*

As previously mentioned, the perspective transform was performed directly after distortion correction. I found it much easier to tune the various color space and gradient transforms when viewing the binary image from the bird’s eye view. Unexpectedly the bird's eye view also created an excellent mask for only the area of interest in the image. This code has been implemented in section `3a. Birds-Eye Perspective Transform` in the `P4-AdvancedLanes-Tuning.ipynb` notebook and the function is called `birds_eye()`.

The birds_eye() function takes as inputs an image (`img`) and harcodes the source (`src`) and destination (`dst`) points. After considerable tuning, mostly experimenting with the transformed image's width and height, the source and destination points were finalized to be:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 578, 460      | 200, 100      | 
| 200, 720      | 200, 720      |
| 705, 460      | 1000, 100     |
| 1140, 720     | 1000, 720     |

I verified that my perspective transform was working as expected by visually inspecting the resulting image for parallel lines when using images with straight lane lines. The before and after results can be seen in the following picture:

<img src="images/birds_eye.png" width="1000">

#### 4. Lane Line Curve Fitting (Histograms and Sliding Windows)

*Rubric: Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?*

After creating the `binary_warped` image, the next step was to apply a histogram to the image to determine the peak locations of pixels in the image. A function to determine the histogram is coded in section `Step 4a. Histogram of Output` in the notebook. Once the histogram was found, a sliding window technique was used to isolate the lane line pixels in the image. This code can be found in section `4b. Sliding Windows and Polynomial Fit` in the notebook. 

This was a critical step in the code for determining the proper left and right lane lines. The sliding windows (rectangles) start with their center at the X locations were the histogram peaks occur. The sliding windows are then extended 100 pixels in the +/- X directions (`margin = 100`) and look for a minimum of 50 pixels. If this condition is met, then the next sliding window will be cetered at the average X location of these pixels, otherwise they will continue to center around the histogram peaks. Once the lane pixel indices and X,Y locations are determined, a 2nd order polynomial fit was used to fit lines to these pixels for both the left and right lane lines using the numpy polyfit function.
```
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

An example of the histogram and the sliding window fit is shown below.

<img src="images/histogram.png" width="400">
<img src="images/sliding_windows.png" width="400">

This next step is where I spent the most time and achieved the biggest gains in results. I used several different techniques to check that the determined polynomial fits for each lane line were realistic. 

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
