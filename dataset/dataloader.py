from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from configs.train_config import DATA_DIR, BATCH_SIZE, NUM_WORKERS
from dataset.dataset import Xray

if __name__ == '__main__':
    train_data = Xray(root=DATA_DIR,train = True, transform=ToTensor())

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True
    )