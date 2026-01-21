from argparse import ArgumentParser

import numpy as np
import torch
import cv2
from torch import nn

from torchsummary import summary

from configs.train_config import IMG_SIZE
from models.CNN_model import CNN_model
from models.resnet import ResNet


def get_args():
    parser = ArgumentParser(description='CNN Training')
    parser.add_argument("--image_size","-i", type=int, default=IMG_SIZE, help="Number of image size")
    parser.add_argument("--image_path","-p", type=str, default="", help="Path image")
    parser.add_argument("--checkpoint","-c", type=str, default="trained_model/best_cnn.pt", help="checkpoint")
    parser.add_argument("--model_type","-m", type=str, default="resnet", help="Type of model")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.model_type == "resnet":
        model = ResNet().to(device)
    else:
        model = CNN_model().to(device)

    categories = ["NORMAL", "PNEUMONIA"]

    # summary(model, input_size=(3, args.image_size, args.image_size))
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
    else:
        print("No checkpoint file found")
        exit(0)

    model.eval()
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1))/255.0
    image = image[None,:,:,:] # Tensor 4 chiều 1x2x224x224
    image = torch.from_numpy(image).to(device).float()
    softmax=nn.Softmax()

    with torch.no_grad():
        output = model(image)
        print(output)
        probs = softmax(output)
        print(probs)
    max_idx = torch.argmax(probs) # or output
    predicted_class = categories[max_idx]
    print(predicted_class)

    # cv2.imshow("{}:{:.2f}%".format(predicted_class, probs[0, max_idx]),ori_image)
    show_img = cv2.resize(ori_image, (800, 600))
    cv2.imshow("{}:{:.2f}%".format(predicted_class, probs[0, max_idx]), show_img)

    cv2.waitKey(0)