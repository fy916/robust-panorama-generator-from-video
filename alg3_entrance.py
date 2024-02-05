import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from alg3_sky_renderer.networks import *
from alg3_sky_renderer.skyboxengine import *
from alg3_sky_renderer import utils
import torch



# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='SKYAR')

# by default, we use this configuration, which will be updated by updateJson function in  alg_processing_utils.py
args = utils.parse_config(path_to_json='./alg3_sky_renderer/config/config-annarbor-castle.json')


class SkyFilter():

    def __init__(self, args, w, h):

        self.ckptdir = args.ckptdir
        self.datadir = args.datadir
        self.input_mode = args.input_mode

        self.in_size_w, self.in_size_h = args.in_size_w, args.in_size_h
        self.out_size_w, self.out_size_h = args.out_size_w, args.out_size_h

        self.out_size_w, self.out_size_h = w, h
        self.skyboxengine = SkyBox(args)

        self.net_G = define_G(input_nc=3, output_nc=1, ngf=64, netG=args.net_G).to(device)
        self.load_model()

        self.video_writer = cv2.VideoWriter('./workfolder/demo.avi',
                                            cv2.VideoWriter_fourcc(*'MJPG'),
                                            20.0,
                                            (args.out_size_w, args.out_size_h))
        self.video_writer_cat = cv2.VideoWriter('./workfolder/demo-cat.avi',
                                                cv2.VideoWriter_fourcc(*'MJPG'),
                                                20.0,
                                                (2 * args.out_size_w, args.out_size_h))

        if os.path.exists(args.output_dir) is False:
            os.mkdir(args.output_dir)

        self.output_img_list = []
        self.result_list= []

        self.save_jpgs = args.save_jpgs

    def load_model(self):
        # load pretrained sky matting model
        print('loading the best checkpoint...')
        checkpoint = torch.load(os.path.join(self.ckptdir, 'best_ckpt.pt'),
                                map_location=device)
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.net_G.to(device)
        self.net_G.eval()

    def write_video(self, img_HD, syneth):

        frame = np.array(255.0 * syneth[:, :, ::-1], dtype=np.uint8)
        self.video_writer.write(frame)
        self.result_list.append(frame)

        frame_cat = np.concatenate([img_HD, syneth], axis=1)
        frame_cat = np.array(255.0 * frame_cat[:, :, ::-1], dtype=np.uint8)
        self.video_writer_cat.write(frame_cat)

        # define a result buffer
        self.output_img_list.append(frame_cat)

    def synthesize(self, img_HD, img_HD_prev):

        h, w, c = img_HD.shape

        img = cv2.resize(img_HD, (self.in_size_w, self.in_size_h))

        img = np.array(img, dtype=np.float32)
        img = torch.tensor(img).permute([2, 0, 1]).unsqueeze(0)

        with torch.no_grad():
            G_pred = self.net_G(img.to(device))
            G_pred = torch.nn.functional.interpolate(G_pred,
                                                     (h, w),
                                                     mode='bicubic',
                                                     align_corners=False)
            G_pred = G_pred[0, :].permute([1, 2, 0])
            G_pred = torch.cat([G_pred, G_pred, G_pred], dim=-1)
            G_pred = np.array(G_pred.detach().cpu())
            G_pred = np.clip(G_pred, a_max=1.0, a_min=0.0)

        skymask = self.skyboxengine.skymask_refinement(G_pred, img_HD)

        syneth = self.skyboxengine.skyblend(img_HD, img_HD_prev, skymask)

        return syneth, G_pred, skymask

    def cvtcolor_and_resize(self, img_HD):

        img_HD = cv2.cvtColor(img_HD, cv2.COLOR_BGR2RGB)
        img_HD = np.array(img_HD / 255., dtype=np.float32)
        img_HD = cv2.resize(img_HD, (self.out_size_w, self.out_size_h))

        return img_HD

    def process_video(self):

        # process the video frame-by-frame

        cap = cv2.VideoCapture(self.datadir)
        m_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_HD_prev = None

        for idx in range(m_frames):
            ret, frame = cap.read()
            if ret:
                img_HD = self.cvtcolor_and_resize(frame)

                if img_HD_prev is None:
                    img_HD_prev = img_HD

                syneth, G_pred, skymask = self.synthesize(img_HD, img_HD_prev)

                # this function saves the image to video file and also saves in the class for us to return results
                self.write_video(img_HD, syneth)

                img_HD_prev = img_HD

                if (idx + 1) % 50 == 0:
                    print(f'processing video, frame {idx + 1} / {m_frames} ... ')

            else:  # if reach the last frame
                break
        return self.result_list[1]

def run_sky_replacement(w, h):
    # get the model object
    sf = SkyFilter(args, w, h)
    image = sf.process_video()
    return image