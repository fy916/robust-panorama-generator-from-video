'''
Author: Feiyang Wang
This is the main entrance of this coursework
Please see the README file for more information
Directly run of this code can generate the result on the report
'''


# import libraries
import cv2

# import implemented and modified code
from alg2A_entrance import run_lowlight_A
from alg2B_entrance import run_lowlight_B
from alg3_entrance import run_sky_replacement
from alg_pano_generation import generate_pano, show
from alg_processing_utils import stablization_first_pass, stablization_second_pass, directional_correction, video_rendering, \
    video_reader, updateJson, get_frame_wh





# select your video file here
input_file_name = "./test_video/video1.mp4"



'''
PRE PROCESSING STAGE

# the following code is used for preprocessing input video if the video is not well recorded
# i.e. the recording is moving left and right without a fixed direction; 
# or the recording needs to be stabilized
# if you want to skip this progress, simply change the pre_processing_output_file_name to your video file name
# to directly process the video to generate panorama
'''
pre_processing_temp_file_name = "workfolder/pre_processing_tmp.mp4"
pre_processing_output_file_name = "workfolder/pre_processing_output.mp4"
###############################################################################
frames = video_reader(input_file_name)
processed_frames, scene_changes = stablization_first_pass(input_file_name, pre_processing_temp_file_name)
sorted_frames = directional_correction(processed_frames, scene_changes)
video_rendering(sorted_frames, pre_processing_temp_file_name)
stablization_second_pass(pre_processing_temp_file_name, pre_processing_output_file_name)
###############################################################################





'''
PANORAMA GENERATION STAGE

I have implemented a stitching method in the file merge_pano.py, please check that for details.

The code here skips the low light enhancement process, if you want to use low light enhancement, please change 
the parameter of pre_processing_output_file_name to low_light_enhancement_output_file_name for the generate_pano function.
'''
output_pano_image_name = "pano.png"
pano = generate_pano(pre_processing_output_file_name, output_pano_image_name)
print("Panorama Image Successfully Saved to ", output_pano_image_name)




'''
LOW LIGHT ENHANCEMENT STAGE

# the following code is used for low light enhancement, only use when video is dark, please pick one you want to use
# if you want to use dual solution from alg2_low_light_A, set lime parameter to False
'''
###############################################################################
img = cv2.imread(output_pano_image_name)

low_light_enhancement_output_file_name = "low_light_output_A.png"
result_frame = run_lowlight_A(img, lime = "True")
cv2.imwrite(low_light_enhancement_output_file_name, result_frame)
print("Low Light Enhanced Image Successfully Saved to ", low_light_enhancement_output_file_name)

low_light_enhancement_output_file_name = "low_light_output_B.png"
result_frame = run_lowlight_B(img)
cv2.imwrite(low_light_enhancement_output_file_name, result_frame)
print("Low Light Enhanced Image Successfully Saved to ", low_light_enhancement_output_file_name)
###############################################################################







'''
SKY REPLACEMENT STAGE

Code here by default changes the sky and save it as another image file.

Please note, sometimes there will be lags when Python reads files once it has the cache, thus even though the json is 
updated the program may no read the updated file immediately.

If the program produces error, please just run it again without needing to change any settings, it should work well. 
'''
output_skychange_image_name = "pano_sky_changed.png"
sky_file_name = "mysky.jpg"

img = cv2.imread(output_pano_image_name)
video_rendering([img, img, img], "./alg3_sky_renderer/test_videos/result.mp4")
w, h = get_frame_wh([img])
updateJson(w, h, sky_file_name)
final_result = run_sky_replacement(w, h)
cv2.imwrite(output_skychange_image_name, final_result)
print("Sky Replacement Image Successfully Saved to ", output_skychange_image_name)

