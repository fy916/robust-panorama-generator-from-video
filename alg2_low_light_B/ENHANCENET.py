import time, itertools

from matplotlib import pyplot as plt

from .dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from .networks import *
from .utils import *
from glob import glob
from PIL import Image
from cv2 import resize
def show(img):
    img = cv2.convertScaleAbs(img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
class ENHANCENET(object) :
    def __init__(self, args, frame_lists_for_processing):
        self.model_name = 'ENHANCENET'
        self.frame_lists_for_processing = frame_lists_for_processing
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.datasetpath = args.datasetpath
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        self.adv_weight = args.adv_weight
        self.identity_weight = args.identity_weight
        self.atten_weight = args.atten_weight
        self.use_gray_feat_loss = args.use_gray_feat_loss
        if args.use_gray_feat_loss == True:
            self.feat_weight = args.feat_weight

        self.n_res = args.n_res
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        self.im_suf_A = args.im_suf_A

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True
        print("# datasetpath : ", self.datasetpath)

    def build_model(self):
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
      
        self.testA = ImageFolder(os.path.join(self.datasetpath), self.test_transform)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

    def load(self, dir, step):
        params = torch.load(r"alg2_low_light_B/results/LOL/model/LOL_params_0900000.pt", map_location=torch.device(self.device))
        self.genA2B.load_state_dict(params['genA2B'])
        self.disGA.load_state_dict(params['disGA'])
        self.disLA.load_state_dict(params['disLA'])

    def test(self):
        print("alg2_low_light_B/results/LOL/model/LOL_params_0900000.pt")
        model_list = glob("alg2_low_light_B/results/LOL/model/LOL_params_0900000.pt")
        if not len(model_list) == 0:
            model_list.sort()
            print('model_list',model_list)
            for i in range(-1,0,1):
                iter = int(model_list[i].split('_')[-1].split('.')[0])
                print('iter',iter)
                self.load("alg2_low_light_B/results/LOL/model/", iter)
                print(" Load SUCCESS")

                self.genA2B.eval()
                 
                path_fakeB=os.path.join('./workfolder')
                if not os.path.exists(path_fakeB):
                    os.makedirs(path_fakeB)

                # self.gt_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(self.datasetpath)) if f.endswith(self.im_suf_A)]




                processed_frames = []

                for frame in self.frame_lists_for_processing:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    show(frame)

                    # create PIL Image object
                    img = Image.fromarray(frame)
                    # for n, img_name in enumerate(self.gt_list):
                    #     print('predicting: %d / %d' % (n + 1, len(self.gt_list)))
                    
                    # img = Image.open(os.path.join(self.datasetpath,  img_name + self.im_suf_A)).convert('RGB')
                    img_width, img_height =img.size
                    
                    real_A = (self.test_transform(img).unsqueeze(0)).to(self.device)
                                            
                    fake_A2B, _, _ = self.genA2B(real_A)
                    
                    A_real = RGB2BGR(tensor2numpy(denorm(real_A[0])))

                    B_fake = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
                    A_real = resize(A_real, (img_width, img_height))
                    B_fake = resize(B_fake, (img_width, img_height))

                    result_frame = cv2.cvtColor(B_fake*255.0, cv2.COLOR_RGB2BGR)
                    processed_frames.append(result_frame.astype(int))
                    cv2.imwrite(os.path.join(path_fakeB,  '%s_low_light_B.png' ), B_fake * 255.0)
                return processed_frames