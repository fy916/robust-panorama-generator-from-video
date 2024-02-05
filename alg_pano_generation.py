'''
Author: Feiyang Wang
This file contains the self-implemented panorama stitching algorithm based on the modification of the following:
stitching: https://blog.51cto.com/u_15483653/4939597
mask generation: https://github.com/tranleanh/image-panorama-stitching/blob/master/panorama.py
cylindrical warpping pre-processing: https://gist.github.com/royshil/0b21e8e7c6c1f46a16db66c384742b2b
'''

from matplotlib import pyplot as plt
import cv2
import numpy as np
import cv2 as cv
import imutils


# read in frames from video file
def read_frames(video_path='./workfolder/pre_processing_output.mp4'):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        newframe = cv2.resize(frame, (720, 1080))
        frames.append(newframe)
    return frames


# show image using matplotlib, easy for debugging and illustration
def show(img):
    img = cv2.convertScaleAbs(img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# the cylindrical warping method from  https://gist.github.com/royshil/0b21e8e7c6c1f46a16db66c384742b2b
# modifications are done to fine tune the mock and warpping degree that is best for iPhone 12 Pro 26mm back camera
def cylindricalWarp(img):
    h, w = img.shape[:2]
    K = np.array([[1030, 0, w / 2], [0, 1030, h / 2], [0, 0, 1]])  # mock intrinsics
    h_, w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_, w_))
    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homog
    Kinv = np.linalg.inv(K)
    X = Kinv.dot(X.T).T  # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w_ * h_, 3)
    B = K.dot(A.T).T  # project back to image-pixels plane
    # back from homog coords
    B = B[:, :-1] / B[:, [-1]]
    # make sure warp coords only within image bounds
    B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[:, 1] < 0) | (B[:, 1] >= h_)] = -1
    B = B.reshape(h_, w_, -1)
    return cv2.remap(img, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA,
                     borderMode=cv2.BORDER_TRANSPARENT)


# utils for cropping the frame, not used for final implementation
def crop_frame(frame):
    height, width = frame.shape[:2]
    crop_width_A = int(0.05 * width)
    crop_width_B = int(0.95 * width)
    cropped_frame = frame[:, crop_width_A:crop_width_B]
    return cropped_frame


# resize the frame to 1.1x and crop the black part which is caused by the cylindrical warpping
# this function keeps the original image size
def resize_frame(frame):
    # Get the current size of the image
    h, w = frame.shape[:2]

    # Calculate the new size of the image
    new_h, new_w = int(h * 1.1), int(w * 1.1)
    padding_up = int((new_h - h) / 2)
    padding_left = int((new_w - w) / 2)
    padding_down = int(padding_up + h)
    padding_right = int(padding_left + w)
    # Resize the image to the new size using INTER_AREA interpolation method
    resized_img = cv2.resize(frame, (new_w, new_h))
    resized_img = resized_img[padding_up:padding_down, padding_left:padding_right, ]
    return resized_img




# I have implemented the panorama generation for both generating from left and right
# so that we can start from the middle and stitch left and right consequetively
class Stitcher_pano_left:
    def __init__(self):
        self.isv3 = imutils.is_cv3()
        self.sift = cv2.SIFT_create()

    # create a mask for blending so that the edge of two blended images will be smooth
    def create_mask(self, img1, img2, widthA, widthB, version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        # after fine-tuning 50 works well
        windowsize = 50

        # for the image on the left, which is new image
        if version == 'left':
            offset = int(windowsize / 2)
            # here the feathering only happens on the inner left side of panorama, in case the newly appended image
            # does not have new information on the left
            barrier = widthB + offset
            mask = np.zeros((height_img1, width_img1))
            # generate a linear mask for the left so that the opacity will gradually reduce to 0 from left to right
            # the opacity gets 0 when it is the right-most of the image content
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_img1, 1))
            mask[:, :barrier - offset] = 1

        # for the image on the right, which is panorama
        else:
            offset = int(windowsize / 2)
            barrier = widthB + offset
            mask = np.zeros((height_img1, width_img2))
            # generate a linear mask for the right so that the opacity will gradually increase to 0 from left to right
            # the opacity gets 0 when it is the left-most of the image content
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_img1, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    # stitch two images
    def stitch(self, imgs, ratio=0.75, reprojThresh=4.0):
        (img2, img1) = imgs
        (kp1, des1) = self.detectAndDescribe(img1)  # new left image for stitching
        (kp2, des2) = self.detectAndDescribe(img2)  # existing pano
        R = self.matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh) # get a mask of the key points

        if R is None:
            # no matching, directly return the original pano img
            return img2
        (good, M, mask) = R

        # I tried to use warpPerspective + findHomography, but the image will stretch to very long when it keeps appending
        # affine transformation will keep the parallelism and ratios of distances, it seems to be a better tool
        # result = cv.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))

        # warp the new image to fit it to a large canvas on which it is projected to a place
        # where its cooresponding place existing in the panorama will be overlapping
        result = cv.warpAffine(img1, M, (img2.shape[1], img1.shape[0]))

        # generate a mask for both panorama and the new image for a smooth blending
        mask_pano = self.create_mask(img2, result, img2.shape[1], img1.shape[1], "right")
        mask_newImg = self.create_mask(img2, result, img2.shape[1], img1.shape[1], "left")

        # apply the mask
        img2 = img2 * mask_pano
        result = result * mask_newImg

        # blend two images, by adding the new image to the panorama
        for i, col in enumerate(img2):
            for j, val in enumerate(col):
                if (sum(img2[i, j]) > 1):
                    result[i, j] += img2[i, j]

        return result



    def detectAndDescribe(self, img):
        # get the key points and description of the image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # use sift for feature description
        if self.isv3:
            sift = cv.xfeatures2d.SIFT_create()
            (kps, des) = sift.detectAndCompute(img, None)
        else:
            kps, des = self.sift.detectAndCompute(gray, None)

        kps = np.float32([kp.pt for kp in kps])

        return (kps, des)


    # match the points for projection or transformation
    def matchKeyPoints(self, kp1, kp2, des1, des2, ratio, reprojThresh):
        # use brute force KNN matching method to match two features
        matcher = cv.DescriptorMatcher_create('BruteForce')
        matches = matcher.knnMatch(des1, des2, 2)

        # stores points that can be used for warpping processing
        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                good.append((m[0].trainIdx, m[0].queryIdx))

        if len(good) > 4:
            src_pts = np.float32([kp1[i] for (_, i) in good])
            dst_pts = np.float32([kp2[i] for (i, _) in good])

            # as mentioned above, the warpPerspective + findHomography does not work well, so changed to affine warpping
            # (M, mask) = cv.findHomography(src_pts, dst_pts, cv.RANSAC, reprojThresh)
            (M, mask) = cv.estimateAffine2D(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold=5.0)

            return (good, M, mask)

        # if not enough points for matching, return None
        return None

    # crop left 10% and right 10% of the input new frame to avoid too much distortion near the edge
    def crop_result(self, result_img):
        h = result_img.shape[0]
        w = result_img.shape[1]
        cut_off_percent = 0.1
        w_new = int(w * cut_off_percent)
        result_img = result_img[:, w_new:]
        return result_img


# this class does the same thing as Stitcher_pano_left, but it stitches new frame to the right of the existing panorama
class Stitcher_pano_right:
    def __init__(self):
        self.isv3 = imutils.is_cv3()
        self.sift = cv2.SIFT_create()

    # create a mask for blending so that the edge of two blended images will be smooth
    def create_mask(self, img1, img2, widthA, widthB, version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        # after fine-tuning 50 works well
        windowsize = 50

        # for the image on the left, which is panorama
        if version == 'left':
            offset = int(windowsize / 2)
            # here the feathering only happens on the inner right side of panorama, in case the newly appended image
            # does not have new information on the right
            barrier = widthA - offset
            mask = np.zeros((height_img1, width_img1))
            # generate a linear mask for the left so that the opacity will gradually reduce to 0 from left to right
            # the opacity gets 0 when it is the right-most of the image content
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_img1, 1))
            mask[:, :barrier - offset] = 1

        # for the image on the right, which is new image
        else:
            offset = int(windowsize / 2)
            barrier = widthA - offset
            mask = np.zeros((height_img1, width_img2))
            # generate a linear mask for the right so that the opacity will gradually increase to 0 from left to right
            # the opacity gets 0 when it is the left-most of the image content
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_img1, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    # stitch two images
    def stitch(self, imgs, ratio=0.75, reprojThresh=4.0, showMatches=False):
        (img2, img1) = imgs
        (kp1, des1) = self.detectAndDescribe(img1)  # new right image for stitching
        (kp2, des2) = self.detectAndDescribe(img2)  # pano

        R = self.matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh) # get a mask of the key points

        if R is None:
            # no matching, directly return the original pano img
            return None
        (good, M, mask) = R

        # I tried to use warpPerspective + findHomography, but the image will stretch to very long when it keeps appending
        # affine transformation will keep the parallelism and ratios of distances, it seems to be a better tool
        # result = cv.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))

        # warp the new image to fit it to a large canvas on which it is projected to a place
        # where its cooresponding place existing in the panorama will be overlapping
        # here is different to stitching on left, because stitching on right only needs a larger canvas, does not need
        # to move the existing panorama to right as stitching on left
        result = cv.warpAffine(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))

        # generate a mask for both panorama and the new image for a smooth blending
        mask_pano = self.create_mask(img2, result, img2.shape[1], img1.shape[1], "left")
        mask_newImg = self.create_mask(img2, result, img2.shape[1], img1.shape[1], "right")

        # apply the mask
        img2 = img2 * mask_pano
        result = result * mask_newImg

        # blend two images, by adding the new image to the panorama
        for i, col in enumerate(img2):
            for j, val in enumerate(col):
                if (sum(img2[i, j]) > 1):
                    result[i, j] += img2[i, j]
        return result

    def detectAndDescribe(self, img):
        # get the key points and description of the image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # use sift for feature description
        if self.isv3:
            sift = cv.xfeatures2d.SIFT_create()
            (kps, des) = sift.detectAndCompute(img, None)
        else:
            kps, des = self.sift.detectAndCompute(gray, None)

        kps = np.float32([kp.pt for kp in kps])
        return (kps, des)

    # match the points for projection or transformation
    def matchKeyPoints(self, kp1, kp2, des1, des2, ratio, reprojThresh):
        # use brute force KNN matching method to match two features
        matcher = cv.DescriptorMatcher_create('BruteForce')
        matches = matcher.knnMatch(des1, des2, 2)

        # stores points that can be used for warpping processing
        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                good.append((m[0].trainIdx, m[0].queryIdx))


        if len(good) > 4:
            src_pts = np.float32([kp1[i] for (_, i) in good])
            dst_pts = np.float32([kp2[i] for (i, _) in good])

            # as mentioned above, the warpPerspective + findHomography does not work well, so changed to affine warpping
            # (M, mask) = cv.findHomography(src_pts, dst_pts, cv.RANSAC, reprojThresh)
            (M, mask) = cv.estimateAffine2D(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold=5.0)

            return (good, M, mask)
        # if not enough points for matching, return None
        return None

    # crop left 10% and right 10% of the input new frame to avoid too much distortion near the edge
    def crop_result(self, result_img):
        h = result_img.shape[0]
        w = result_img.shape[1]
        cut_off_percent = 0.9
        w_new = int(w * cut_off_percent)
        result_img = result_img[:, :w_new]
        return result_img




# used to call Stitcher_pano_left class object to stitch images to left of panorama
def calc_left(img1, img2):
    # img 1 is new image, img2 is pano
    stitcher = Stitcher_pano_left()
    # stitch
    result = stitcher.stitch([img1, img2])

    # remove blank rows and columns
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    final_result = cv2.convertScaleAbs(final_result)
    final_result = stitcher.crop_result(final_result)
    return final_result


# used for stitching new image to the left, by padding the existing panorama first so that the left part is empty for
# new image to transform and add
def padding_pano(appendedImg, pano):
    w = appendedImg.shape[1]
    # Get the image size
    height, width, channels = pano.shape
    # Define the desired padding size
    left_pad_size = w
    # Create a black rectangle with the desired padding size
    black_rect = np.zeros((height, left_pad_size, channels), dtype=np.uint8)
    # Concatenate the black rectangle and the original image horizontally
    new_img = np.concatenate((black_rect, pano), axis=1)
    return new_img

# used to call Stitcher_pano_right class object to stitch images to right of panorama
def calc_right(img1, img2):
    # img 1 is pano, img2 is new img
    stitcher = Stitcher_pano_right()
    # stitch
    result = stitcher.stitch([img1, img2], showMatches=True)

    # remove blank rows and columns
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    final_result = cv2.convertScaleAbs(final_result)
    final_result = stitcher.crop_result(final_result)
    return final_result


# used to call calc_left to perform stitching
def mergeLeft(image, result):
    # do the fix of barrel distortion first
    frame_new = cylindricalWarp(image)
    # crop the image on left and right side to avoid too much distortion
    frame_new = resize_frame(frame_new)
    # padding the panorama on the left
    result = padding_pano(frame_new, result)
    # stitch images
    result = calc_left(result, frame_new)
    return result


def mergeRight(result, image):
    # do the fix of barrel distortion first
    frame_new = cylindricalWarp(image)
    # crop the image on left and right side to avoid too much distortion
    frame_new = resize_frame(frame_new)
    # stitch images
    result = calc_right(result, frame_new)
    return result


def cut_corners(img):
    # when the final panorama is generated, sometimes it will have small blank parts on the top and buttom, crop it a bit
    h, w = img.shape[:2]
    percent = 0.08
    new_width = int(w * (1 - 2 * percent))
    newheight = int(h * (1 - 2 * percent))

    height_start = int(h * percent)
    width_start = int(w * percent)

    # crop the panorama
    new_img = img[height_start: height_start + newheight, width_start: width_start + new_width]
    return new_img


# use this function to start from the middle point of video frames and stitch frames on its left and right repeatedly
def generate_pano(video_path, output_path):
    print("Generating panorama, please wait, you can check the status on the right window")

    frames = read_frames(video_path)
    length = len(frames)
    mid_index = length // 2
    result = frames[mid_index]
    result = resize_frame(result)
    i = 1

    # starting from the middle point, then stitch frame at index: mid-1, mid+1, mid-2, mid+2 .....until finished
    while mid_index - i >= 0 or mid_index + i < length:
        print(f"Stitching image progress {i * 2} of {length}.")
        if mid_index - i >= 0:
            result = mergeLeft(frames[mid_index - i], result)
        if mid_index + i < length:
            result = mergeRight(result, frames[mid_index + i])
        i += 1
        show(cut_corners(result))

    cv2.imwrite(output_path, cut_corners(result))
