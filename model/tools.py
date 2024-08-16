import numpy as np
import torch
import os
from torchvision.utils import save_image


def tensor2im(input_image, imtype=np.uint8):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

def get_file_path(current_path):
    global total_file, clear_file
    current_list = []
    for next_path in os.listdir(current_path):
        tmp_path = current_path+'/'+next_path
        if os.path.isfile(tmp_path):
            current_list.append(tmp_path)
        else:
            current_list += get_file_path(tmp_path)
    return current_list

def save_image_tensor(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    save_image(input_tensor, filename)

if __name__ == "__main__":
    pass