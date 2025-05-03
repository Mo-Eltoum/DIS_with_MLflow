import os

import cv2
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm
from torchvision.transforms.functional import normalize
from models.isnet import *

if __name__ == "__main__":

    dataset_path = '/PATH/TO/YOUR/DATA'  # Your dataset path
    model_path = '/PATH/TO/YOUR/MODEL.PTH'  # path to model weights
    result_path = '/PATH/TO/OUTPUT/DIRECTORY'  # path to save the results

    net = ISNetDIS()

    # uncomment if you used multi gpu training flag while training your model
    # net = nn.DataParallel(net)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net.to(device)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))


    net.eval()

    im_list = glob(dataset_path + "/*.jpg") + glob(dataset_path + "/*.JPG") + glob(dataset_path + "/*.jpeg") + glob(
        dataset_path + "/*.JPEG") + glob(dataset_path + "/*.png") + glob(dataset_path + "/*.PNG") + glob(
        dataset_path + "/*.bmp") + glob(dataset_path + "/*.BMP") + glob(dataset_path + "/*.tiff") + glob(
        dataset_path + "/*.TIFF")
    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            print("im_path: ", im_path)
            im = io.imread(im_path) / 255
            im_tensor = torch.tensor(im, dtype=torch.float32)
            im_tensor = torch.transpose(torch.transpose(im_tensor, 1, 2), 0, 1)
            im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1, 1, 1])
            im_tensor = torch.unsqueeze(im_tensor, 0)
            image = F.interpolate(im_tensor, size=(1024, 1024))
            im_shp = im.shape[:2]

            if torch.cuda.is_available():
                image = image.cuda()
            result = net(image)
            result = torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result - mi) / (ma - mi)
            im_name = im_path.split('/')[-1].split('.')[0] + '.png'
            result_img_path = os.path.join(result_path, im_name)
            # io.imsave(os.path.join(result_path, im_name + ".png"),
            #           (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
            np_mask = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
            cv2.imwrite(result_img_path, np_mask)
