from lib import *
from loss import SoftDiceLoss, DiceScore
from models import CTDC
from dataset import get_dataloaders
from train import Args, build, test

args = Args(
    root="./datasets/kvasir", 
    epochs=80, 
    batch_size=4, 
    dataset="Kvasir",
    mgpu="false",
    lrs="true",
    lrs_min=1e-6,
    lr = 1e-4,
    type_lr = "StepLR",
    checkpoint_path = "./checkpoint/kvasir/CTDCformer_epoch_backbonePvitB3_Kvasir.pt",
    backbone="PvtB3",
    optim="AdamW"
)

( device, train_dataloader, val_dataloader, test_dataloader,test_dataloader4vis, Dice_loss,
    BCE_loss, perf, model, optimizer, checkpoint, scheduler, loss_fun) = build(args)

if __name__ == "__main__":
    test_measure_mean, test_measure_std = test(
                model, device, test_dataloader, 300, perf,"Test"
            )