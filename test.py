import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from argparse import ArgumentParser
import torch
import cv2
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from configs.test_config import CATEGORIES, CHECKPOINT_PATH, IMAGE_PATH
from configs.train_config import IMG_SIZE
from models.CNN_model import CNN_model


def get_args():
    parser = ArgumentParser(description='X-Ray Pneumonia Classification')
    parser.add_argument("--image_size", "-i", type=int, default=IMG_SIZE)
    parser.add_argument("--image_path", "-p", type=str, default=IMAGE_PATH)
    parser.add_argument("--checkpoint", "-c", type=str, default=CHECKPOINT_PATH)
    return parser.parse_args()


def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CNN_model().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean, std)
    ])
    
    image = Image.open(args.image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
    
    confidence, idx = torch.max(probs, dim=1)
    predicted_class = CATEGORIES[idx.item()]
    
    print(f"\nResult: {predicted_class} ({confidence.item():.2%})")
    
    img = cv2.imread(args.image_path)
    img = cv2.resize(img, (600, 600))
    cv2.imshow(f"{predicted_class} - {confidence.item():.2%}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()