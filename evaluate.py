import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from argparse import ArgumentParser
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from configs.train_config import IMG_SIZE, NUM_WORKERS, BATCH_SIZE, DATA_DIR
from configs.test_config import CATEGORIES, CHECKPOINT_PATH
from dataset.dataset import Xray
from models.CNN_model import CNN_model
from models.resnet import ResNet


def get_args():
    parser = ArgumentParser(description='Evaluate model and show confusion matrix')
    parser.add_argument("--image_size", "-i", type=int, default=IMG_SIZE)
    parser.add_argument("--checkpoint", "-c", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--data_dir", "-d", type=str, default=DATA_DIR)
    parser.add_argument("--batch_size", "-b", type=int, default=BATCH_SIZE)
    parser.add_argument("--model_type", "-m", type=str, default="cnn", choices=["cnn", "resnet"], help="Model type: cnn or resnet")
    return parser.parse_args()


def plot_confusion_matrix(cm, categories, model_name):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, fontsize=11)
    plt.yticks(tick_marks, categories, fontsize=11)
    
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    filename = f'confusion_matrix_{model_name.lower()}.png'
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.show()


def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if args.model_type == "resnet":
        # Load ResNet (frozen backbone is irrelevant for inference)
        model = ResNet(freeze_backbone=False).to(device)
        print("Using ResNet model")
    else:
        model = CNN_model().to(device)
        print("Using CNN model")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    epoch = checkpoint.get('epoch', 'N/A')
    best_acc = checkpoint.get('best_accuracy', 'N/A')
    print(f"Loaded: Epoch {epoch}, Best Acc: {best_acc:.2%}" if isinstance(best_acc, float) else f"Loaded: Epoch {epoch}")
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean, std)
    ])
    
    test_dataset = Xray(root=args.data_dir, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"Test samples: {len(test_dataset)}")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
    
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CATEGORIES))
    
    plot_confusion_matrix(cm, CATEGORIES, args.model_type.upper())


if __name__ == '__main__':
    main()
