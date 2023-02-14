import torch
import torch.nn as nn
from autoencoder import TransNetEncoder, TransNetDecoder, TransNet
from torch import Tensor
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import CIFAR100
from torchvision import transforms, models
from Dataset import CustomDataSet
import time
from tqdm import tqdm
import config
from utils import save_some_examples
import wandb
# wandb.init(project="my-awesome-project")



def train(TransNet_vis, TransNet_ir, optimizer_vis, optimizer_ir, trainloader, criterion, epoch, NUM_EPOCHS):
    train_loss = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # loop = tqdm(trainloader, leave=True)
    running_loss = 0.0
    start = time.time()
    for idx, (x, y) in enumerate(trainloader):
        x = x.to(device)
        y = y.to(device)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            enc_vis = TransNet_vis.encoder(x)
            dec_vis = TransNet_vis.decoder(enc_vis)
            enc_ir = TransNet_ir.encoder(y)
            dec_ir = TransNet_ir.decoder(enc_ir)
            vis_to_ir = TransNet_ir.decoder(enc_vis)
            ir_to_vis = TransNet_vis.decoder(enc_ir)
            L_recon = criterion(x, dec_vis) + criterion(y, dec_ir)
            L_trans = criterion(x, ir_to_vis) + criterion(y, vis_to_ir)
            loss = L_recon + L_trans
            optimizer_vis.zero_grad()
            optimizer_ir.zero_grad()
            loss.backward()
            optimizer_vis.step()
            optimizer_ir.step()
            running_loss += loss.item()
            # wandb.log({'epoch': epoch, 'loss': running_loss})
        loss = running_loss / len(trainloader)

        train_loss.append(loss)
        end = time.time()
        total_time = end - start
        print('- Epoch {} of {}, ETA: {:.2f} Train Loss: {:.5f}'.format(
            epoch+1, NUM_EPOCHS, total_time, loss))
        
        # if epoch % 5 == 0:
        #     save_decoded_image(img.cpu().data, name='original{}'.format(epoch))
        #     save_decoded_image(outputs.cpu().data, name='decoded{}'.format(epoch))
    return train_loss

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # Assuming that we are on a CUDA machine, this should print a CUDA device:

# print(device)
def main():
    transform = transforms.Compose([transforms.Resize((512,512),transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])
    train_dataset = CustomDataSet(config.TRAIN_DIR_VIS, config.TRAIN_DIR_IR, transform= transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=config.NUM_WORKERS)
    criterion = nn.L1Loss()
    TransNet_vis = TransNet().to(config.DEVICE)
    TransNet_ir = TransNet().to(config.DEVICE)
    optimizer_vis = optim.Adam(TransNet_vis.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_ir = optim.Adam(TransNet_ir.parameters(), lr = config.LEARNING_RATE, betas = (0.5, 0.999))
    val_dataset = CustomDataSet(config.VAL_DIR_VIS, config.VAL_DIR_IR, transform= transform)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=config.NUM_WORKERS)
    for epoch in range(config.NUM_EPOCHS):
        train(TransNet_vis, TransNet_ir, optimizer_vis, optimizer_ir, train_loader, criterion, epoch, config.NUM_EPOCHS)
        if epoch % 1 == 0 :
            save_some_examples(TransNet_vis, TransNet_ir, val_loader, epoch, 'evaluation')

if __name__ == "__main__":
    main()