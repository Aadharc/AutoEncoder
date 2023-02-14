# just a comment

import torch
import config
from torchvision.utils import save_image

def save_some_examples(TransNet_vis, TransNet_ir, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    TransNet_ir.eval()
    TransNet_vis.eval()
    with torch.no_grad():
        # y_fake = gen(x)
        # y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        # y_fake = y_fake 
        # 
        enc_vis = TransNet_vis.encoder(x)
        dec_vis = TransNet_vis.decoder(enc_vis)
        enc_ir = TransNet_ir.encoder(y)
        dec_ir = TransNet_ir.decoder(enc_ir)
        vis_to_ir = TransNet_ir.decoder(enc_vis)
        ir_to_vis = TransNet_vis.decoder(enc_ir) 
        save_image(dec_ir, folder + f"/dec_ir_{epoch}.png")
        save_image(dec_vis, folder + f"/dec_vis_{epoch}.png")
        save_image(vis_to_ir, folder + f"/vis_to_ir_{epoch}.png")
        save_image(ir_to_vis, folder + f"/ir_to_vis_{epoch}.png")
        # save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        save_image(x, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y, folder + f"/label_{epoch}.png")
            # save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    TransNet_ir.train()
    TransNet_vis.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
