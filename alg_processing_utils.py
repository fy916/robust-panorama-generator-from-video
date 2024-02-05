'''
Author: Feiyang Wang
This file contains some utilities used in this project
'''
import copy
from alg1_video_stab import VidStab
import cv2




# output the video of a list to file
def video_rendering(frames, filename = "tmp.mp4"):
    h, w = frames[0].shape[0], frames[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, fourcc, 30.0,(w, h))

    # Define the list of frames
    frames = frames

    # Loop over the frames and write each frame to the video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object and close the file
    out.release()


# read the video from file and return a frame list
def video_reader(filename = "input.mp4"):
    # Open the video file
    cap = cv2.VideoCapture(filename)
    # Initialize an empty list to store the frames
    frames = []
    # Loop through the video frames
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        # If the frame was not read correctly, break the loop
        if not ret:
            break
        # Append the frame to the list
        frames.append(frame)
    # Release the video capture object
    cap.release()
    return frames




# this function is about stablizating using the modified stablization library in alg1_video_stab
def stablization_first_pass(file_path = 'input.mp4', output_path = "tmp.mp4"):
    print("Doing First Stabilization, Please Wait! ")
    stabilizer = VidStab(kp_method='FAST', threshold=32, nonmaxSuppression=False)
    processed_frames = stabilizer.stabilize(input_path=file_path,
                                            smoothing_window=5,
                                            output_path=output_path,
                                            show_progress=True)

    # get the coordinate changes of videos from stabilization process for further analysis
    scene_changes = stabilizer.smoothed_trajectory
    stabilizer.plot_trajectory()

    return processed_frames, scene_changes



# this function will generate a list containing sampled frames from video
# which may have different moving speed, multiple directions, and different lengths
def directional_correction(processed_frames, scene_changes):
    print("Doing Directional Correction! Please Wait ")
    x_corrds = []

    # save the x coordinates of each frame in a new list
    for val in scene_changes:
        valX = val[0]
        x_corrds.append(valX)


    # Generate the indices list by creating a range object
    indices = list(range(len(x_corrds)))

    # Sort the indices based on the corresponding values (in descending order)
    indices.sort(key=lambda i: x_corrds[i], reverse=True)
    sorted_indices = [int(i) for i in indices]


    max_x_idx = sorted_indices[0]
    min_x_idx = sorted_indices[-1]


    sorted_frames = []
    processed_frames # original frames
    sorted_indices # x sorted index
    x_corrds # x values
    kepted_idx = []

    # The following code gets the intermediate frames between left-most frame to right-most frame (remove duplication)
    # if the left most frame appears before the right-most frame
    if max_x_idx < min_x_idx:
        sorted_frames.append(processed_frames[max_x_idx])
        curr_idx = max_x_idx
        kepted_idx.append(curr_idx)
        for idx in range(max_x_idx, min_x_idx+1):
            # if this frame is on the right of last marked frame by change of 50, append it to use
            if x_corrds[idx] < x_corrds[curr_idx] - 50:
                sorted_frames.append(processed_frames[idx])
                curr_idx = idx
                kepted_idx.append(curr_idx)


    # if the left most frame appears after the right-most frame
    else:
        sorted_frames.append(processed_frames[min_x_idx])
        curr_idx = min_x_idx
        kepted_idx.append(curr_idx)
        for idx in range(min_x_idx, max_x_idx+1):
            # if this frame is on the left of last marked frame by change of 50, append it to use
            if x_corrds[idx] > x_corrds[curr_idx] + 50:
                sorted_frames.append(processed_frames[idx])
                curr_idx = idx
                kepted_idx.append(curr_idx)
        sorted_frames = copy.deepcopy(sorted_frames[::-1])
        kepted_idx =  copy.deepcopy(kepted_idx[::-1])

    return sorted_frames


# this function is about stablizating using the modified stablization library in alg1_video_stab
def stablization_second_pass(input_path = "directional_corrected.mp4", output_path = "tmp.mp4"):
    print("Doing Stablization on Sampled Frames, Please Wait! ")

    stabilizer = VidStab(kp_method='FAST', threshold=32, nonmaxSuppression=False)
    processed_frames = stabilizer.stabilize(input_path=input_path,
                                            smoothing_window=5,
                                            output_path=output_path,
                                            show_progress=True)

    scene_changes = stabilizer.smoothed_trajectory

    stabilizer.plot_trajectory()

    import matplotlib.pyplot as plt
    plt.show()



# check if the orientation of the video is vertical
def check_orientation_vertical(input_frames):
    if len(input_frames)>0:
        if input_frames[0].shape[0] > input_frames[0].shape[1]:
            return True
        else:
            return False
    else:
        return None


# update the json file for alg3_sky_renderer
def updateJson(w, h, sky_file):
    import json

    # open the JSON file for reading
    with open('./alg3_sky_renderer/config/config-annarbor-castle.json', 'r') as f:
        config = json.load(f)

    # modify the out_size_w and out_size_h values
    config["out_size_w"] = w
    config["out_size_h"] = h
    config["skybox"] = sky_file

    # open the JSON file for writing
    with open('./alg3_sky_renderer/config/config-annarbor-castle.json', 'w') as f:
        json.dump(config, f, indent=4)

    import time
    time.sleep(5) # wait for saving update


# get the frame width and height
def get_frame_wh(frames):
    if len(frames)>0:
        h = frames[0].shape[0]
        w = frames[0].shape[1]
        return w, h
    else:
        return None
