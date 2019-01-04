# stereo-view-synthesis
  Generate novel view image from image pairs
# Dataset
  The referenced dataset is invented and described in the paper "Stereo Magnification: Learning View Synthesis using Multiplane Images. Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, Noah Snavely. SIGGRAPH 2018".  
  Please refer to https://github.com/google/stereo-magnification and https://google.github.io/realestate10k/ for details.  
    
  Each image corresponds to a set of camera parameters.  
  The images are frames in videos and are not rectified.  
    
  *image_data_example.zip* and *camera_para_example.txt* give an example of 55 images obtained from 80 timestamps in *camera_para_example.txt* (due to file size limit, the last 25 images are not uploaded).  
    
  Image frames are obtained using opencv with the given timestamps and the videos are downloaded using pytube.  
    
  *get_frame.py* provides a script for getting the image frames.  
    
  Full downloaded image dataset can be find in 集群 10.1.75.35, */panzheng/dataset/stereo_dataset*:  
            */test_image*: files containg valid images for testing (test/train categorized by original author)  
            */train_image*: files containg valid images for training (test/train categorized by original author)  
            */valpos_test*: valid poses for testing (.txt)  
            */valpos_train*: valid poses for training (.txt)  
            */train_image_i0*: only contains the first image file for training  
            */valpos_train_i0*: only contains valid poses for images in */train_image_i0*  
      
  Please notice that some video urls provided in https://google.github.io/realestate10k/ may be invalid, and some timestamps may not yield valid images. The camera parameter txts in the above files are modified for validity by Pan Zheng.  
  */test_image* matches with */valpos_test*, */train_image* mathces with */valpos_train*. The images and camera pose txts are named sequentially. 1 single file contained in */test_image* or */train_image* has around 200000 image frames.  
    
# Training the model
  The python script for training is *train_view_syn.py*.  
    
  The model *1)* gets image pairs and camera parameters provided by the dataset as input, *2)* computes the disparity using network provided by Liusheng (tf-model file), *3)* converts disparity to depth, *4)* warps the depth to desired view, *5)* forward (or backward) warps the source images to the target view with corresponding camera matrices, *5)* concatenates source images + warped images + disparity + camera poses, *6)* feeds the concatenated channels together with the groundtruth image into the network, and *7)* outputs the image in novel view. L1 loss is implemented.  
  The network consists of a 15 layers encoder-decoder structure followed by 2 layers of 2d convolution.  
  
 # Testing the model
   The python script for testing is *test_view_syn.py*.  
   One may input imageL, imageR, poseL, poseR, pose_target and obtain an image in the view of pose_target.  
     
   The restored trained model example can be found in the file trained_imperfect_model. The presented model can be found in the file *model-2000iter*. Due to the large amount of dataset images and time limit, the presented model was only trained with 1480 sets of training images (2 source images + 1 target image) for 2000 iterations with learning rate 1e-4 and batch size 8.  
     
   An example tested result is in *result_imperfect.zip*. The testing frames are two images in the test image example *image_data_example.zip*. One can observe that although the rendered image is blurry, the position fits the desired novel view quite well.  
     
 # Notification
   Please notice that *train_view_syn.py* and *test_view_syn.py* should be put in the directory /tf-model for successful running.  
