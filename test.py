from config import *
from glob import glob
from src.model import *
from src.camvid_pipeline import *
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage



train_images = sorted(glob('/home/ubuntu/vision-benchmark-datasets/camvid/train/*'))
train_labels = sorted(glob('/home/ubuntu/vision-benchmark-datasets/camvid/trainannot/*'))
val_images = sorted(glob('/home/ubuntu/vision-benchmark-datasets/camvid/val/*'))
val_labels = sorted(glob('/home/ubuntu/vision-benchmark-datasets/camvid/valannot/*'))
test_images = sorted(glob('/home/ubuntu/vision-benchmark-datasets/camvid/test/*'))
test_labels = sorted(glob('/home/ubuntu/vision-benchmark-datasets/camvid/testannot/*'))

train_dataset = CamVidDataset(train_images, train_labels, 512, 512)
val_dataset = CamVidDataset(val_images, val_labels, 512, 512)
test_dataset = CamVidDataset(test_images, test_labels, 512, 512)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

class_weights = get_class_weights(train_loader, 12)

enet = Enet(12)
enet = enet.to(device)
print(enet)

criterion = CrossEntropyLoss()
optimizer = Adam(
    enet.parameters(),
    lr=5e-4,
    weight_decay=2e-4
)

train_loss_history, val_loss_history = train(
    enet, train_loader, val_loader,
    device, criterion, optimizer,
    len(train_images) // 16,
    len(val_images) // 16, 5,
    './checkpoints', 'enet-model', 50
)

plt.plot(train_loss_history, color = 'b', label = 'Training Loss')
plt.legend()
plt.savefig('./plots/plot-camvid-training-loss-{}-epochs.png'.format(50))

plt.plot(val_loss_history, color = 'b', label = 'Validation Loss')
plt.legend()
plt.savefig('./plots/plot-camvid-val-loss-{}-epochs.png'.format(50))