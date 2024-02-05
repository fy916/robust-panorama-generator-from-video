# Panorama Generator
# Author: Feiyang Wang

# File Organization & Introduction


```
.
├── alg1_video_stab
│   └── ...
├── alg2_low_light_A
│   └── ...
├── alg2_low_light_B
│   └── ...
├── alg3_sky_renderer
│   └── ...
│
├── README.md
├── main.py
├── alg_pano_generation.py
├── alg_processing_utils.py
├── alg2A_entrance.py
├── alg2B_entrance.py
├── alg3_entrance.py
├── requirements.txt
│
├── test_video
│   ├── video1.mp4
│   ├── video2.mp4
│   ├── video3.mp4
│   └── ....mp4
├── workfolder
│   └── ...
├── output
│   └── ...
└── result
    └── ...
```

- [main.py](main.py) file contains the main entrance of the program. It contains different parts that triggers running the video stabilization, panorama generation, low light enhancement, night enhancement, and sky changing algorithms.
* ⭐️ [alg_pano_generation.py](alg_pano_generation.py) file contains the main code used for panorama generation. It works by firstly extracting useful frames from lengthy video, then applying cylindrical warping to the frames based on lens parameters, finally stitching from the mid-frame working on a left-right-left-right manner. The proposed panorama method performs well in multiple situations and can handle different video types. The developed methodology outperformed the OpenCV CV2.STITCHER PANORAMA method as it generates high-quality results even for extreme cases of shaky video with varied movement direction patterns that cause the CV2 method to fail. For detailed explanation of implementation details and ideas, please refer to the [report.pdf](report.pdf) or see the end of this readme file.
- [alg_processing_utils.py](alg_processing_utils.py) file contains the utility function for detecting video moving trajectory, applying directional correction for stabilization, updating json configuration, loading video, writing video, etc.
- [alg1_video_stab](alg1_video_stab) folder contains the modified video stabilizer from project [python_video_stab](https://github.com/AdamSpannbauer/python_video_stab). It is tested that by applying stabilization to the given video can immensely improve the performance and the result of panorama stitching.
- [alg2_low_light_A](alg2_low_light_A) folder contains the modified low light enhancement algorithm from project [Low-light-Image-Enhancement](https://github.com/pvnieo/Low-light-Image-Enhancement).
- [alg2_low_light_B](alg2_low_light_B) folder contains the modified night enhancement algorithm from project [night-enhancement](https://github.com/jinyeying/night-enhancement).
- [alg3_sky_renderer](alg3_sky_renderer) folder contains the modified sky changing algorithm from project [SkyAR](https://github.com/jiupinjia/SkyAR).
- [alg2A_entrance.py](alg2A_entrance.py) file contains functions that trigger running the low light enhancement algorithm.
- [alg2B_entrance.py](alg2B_entrance.py) file contains functions that trigger running the night enhancement algorithm.
- [alg3_entrance.py](alg3_entrance.py) file contains functions that trigger running the sky changing algorithm.

# Before Running

As it is difficult to upload very large models (together about 1GB) to moodle (limitation of 10MB) together with this submission, please download the following files and put them into respective locations.



1. (For night low light enhancement) Download checkpoints_G_coord_resnet50.zip from [Google Drive](https://drive.google.com/uc?id=1COMROzwR4R_7mym6DL9LXhHQlJmJaV0J), unzip it, and then put the best_skpt.pt into folder `\alg3_sky_renderer\checkpoints_G_coord_resnet50\`
2. (For sky changing) Download the LOL_params_0900000.pt from [Dropbox - LOL_params_0900000.pt](https://www.dropbox.com/s/0ykpsm1d48f74ao/LOL_params_0900000.pt?dl=0), unzip it, and put it into `alg2_low_light_B\results\LOL\model\ `

Also, if you would only check about the panorama algorithm and the preprocessing steps, you could simply only run the code of these parts. They should generate satisfying result without the need of downloading the models and installation of libraries such as pytorch.

# How to use

Directly run main.py, you can adjust settings in this file as it consists intuitive configurations and function calls. 

Also, the default sky image for sky replacement is mysky.jpg in `alg3_sky_renderer\skybox`, and you can change it by putting new images to this folder, and change the filename (Filename only! no path needed) of sky_file_name variable on line 104 of file `main.py` . 





## Work Report





![Result-4.jpeg](report_pages/Result-4.jpeg)
![Result-5.jpeg](report_pages/Result-5.jpeg)
![Result-6.jpeg](report_pages/Result-6.jpeg)
![Result-7.jpeg](report_pages/Result-7.jpeg)
![Result-8.jpeg](report_pages/Result-8.jpeg)
![Result-9.jpeg](report_pages/Result-9.jpeg)
![Result-10.jpeg](report_pages/Result-10.jpeg)
![Result-11.jpeg](report_pages/Result-11.jpeg)
![Result-12.jpeg](report_pages/Result-12.jpeg)
![Result-13.jpeg](report_pages/Result-13.jpeg)
![Result-14.jpeg](report_pages/Result-14.jpeg)
![Result-15.jpeg](report_pages/Result-15.jpeg)
![Result-16.jpeg](report_pages/Result-16.jpeg)
![Result-17.jpeg](report_pages/Result-17.jpeg)
![Result-18.jpeg](report_pages/Result-18.jpeg)
![Result-19.jpeg](report_pages/Result-19.jpeg)
![Result-20.jpeg](report_pages/Result-20.jpeg)
![Result-21.jpeg](report_pages/Result-21.jpeg)
![Result-22.jpeg](report_pages/Result-22.jpeg)
![Result-23.jpeg](report_pages/Result-23.jpeg)
![Result-24.jpeg](report_pages/Result-24.jpeg)
![Result-25.jpeg](report_pages/Result-25.jpeg)
![Result-26.jpeg](report_pages/Result-26.jpeg)
![Result-27.jpeg](report_pages/Result-27.jpeg)
![Result-28.jpeg](report_pages/Result-28.jpeg)


# References

Panorama stitching is built based on: 

[使用OpenCV进行图像全景拼接_51CTO博客_全景图像拼接](https://blog.51cto.com/u_15483653/4939597)

Mask generation is modified based on:

[image-panorama-stitching/panorama.py at master · tranleanh/image-panorama-stitching · GitHub](https://github.com/tranleanh/image-panorama-stitching/blob/master/panorama.py)

CylindricalWarping is modified based on:

[Warp an image to cylindrical coordinates for cylindrical panorama stitching, using Python OpenCV · GitHub](https://gist.github.com/royshil/0b21e8e7c6c1f46a16db66c384742b2b)

Alg_1: video_stablization is modified based on:

[GitHub - AdamSpannbauer/python_video_stab: A Python package to stabilize videos using OpenCV](https://github.com/AdamSpannbauer/python_video_stab)

Alg_2_A: low light enhancement is modified based on:

[GitHub - pvnieo/Low-light-Image-Enhancement: Python implementation of two low-light image enhancement techniques via illumination map estimation](https://github.com/pvnieo/Low-light-Image-Enhancement)

Alg_2_B: night_enhancement is modified based on:

[GitHub - jinyeying/night-enhancement: [ECCV2022] "Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression", https://arxiv.org/abs/2207.10564](https://github.com/jinyeying/night-enhancement)

Alg_3: sky changing is modified based on:

[GitHub - jiupinjia/SkyAR: Official Pytorch implementation of the preprint paper &quot;Castle in the Sky: Dynamic Sky Replacement and Harmonization in Videos&quot;, in arXiv:2010.11800.](https://github.com/jiupinjia/SkyAR)
