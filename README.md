# UGV image data collection test 

## Example dataset
The sample dataset containing 50 images is in this folder:\
**CV_test_dataset_50**

## Test related code
The script for evaluating single image is:\
*img_assess.py* \
 Download the example image *1.png* to your local machine, modify the image path in the *img_assess.py* source code accordingly, and run: 
```
Python3 img_assess.py
```
The script for evaluating multiple images is:\
*multiple_img_assess.py* \
Download the **CV_test_dataset_50** dataset to your local machine, modify the image dataset path in the *multiple_img_assess.py* source code accordingly, and run: 
```
Python3 multiple_img_assess.py
```
If you record **.bag** files using a RealSense depth camera, you can use *bag_image_capture.py* to automatically capture 50 images and save them as a dataset. Modify the **.bag** file path in the *bag_image_capture.py* source code accordingly, and run:
```
Python3 bag_image_capture.py
```
