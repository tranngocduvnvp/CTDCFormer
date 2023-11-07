from lib import *
from loss import SoftDiceLoss, DiceScore
from models import CTDC
from dataset import get_dataloaders

def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

 
    img_path = args.root + "/images/*"
    input_paths = sorted(glob.glob(img_path))
    depth_path = args.root + "/masks/*"
    target_paths = sorted(glob.glob(depth_path))
    
    train_dataloader, test_dataloader, val_dataloader, test_dataloader4vis = get_dataloaders(
        input_paths, target_paths, batch_size=args.batch_size
    )

    Dice_loss = SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    TverskyLoss = tgm.losses.TverskyLoss(alpha=0.5, beta=0.5)
    FocalLoss = tgm.losses.FocalLoss(alpha=0.5, gamma=1, reduction='mean')
    Ssim = tgm.losses.SSIM(5, reduction='none')
    Smooth = tgm.losses.InverseDepthSmoothnessLoss()
    loss_fun = {'Dice_loss':Dice_loss, "BCE_loss":BCE_loss, "TverskyLoss":TverskyLoss, "FocalLoss":FocalLoss,\
                "Ssim":Ssim, "Smooth":Smooth}

    perf = DiceScore()

    model = CTDC()
    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)

    #===================== Optimizer ===================================================
    if args.optim == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.optim == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "SparseAdam":
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)
    elif args.optim == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optim == "ASGD":
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optim == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr)
    elif args.optim == "NAdam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr)
    elif args.optim == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    elif args.optim == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == "Rprop":
        optimizer = torch.optim.Rprop(model.parameters(), lr=args.lr)
    #===================================================================================

    if args.lrs == "true":
        if args.type_lr == "LROnP":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optimizer, mode="max", patience=10, factor=0.5, min_lr=args.lrs_min, verbose=True)
        elif args.type_lr == "StepLR":
            print("Using StepLR")
            scheduler = torch.optim.lr_scheduler.StepLR(
                  optimizer, step_size=13, gamma=0.4, verbose=False)
        elif args.type_lr == "MultiStepLR":
            print("Using MultiStepLR")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                  optimizer, milestones=[10, 20, 30, 60], gamma=0.5, verbose=False)

        
    if args.checkpoint_path == None:
        checkpoint = {"val_measure_mean":None, "epoch":0}
    else:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (device, train_dataloader, val_dataloader, test_dataloader, test_dataloader4vis, Dice_loss,
        BCE_loss, perf, model, optimizer, checkpoint, scheduler, loss_fun)




def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        for k in range(0, data.shape[0], 4):
            data_input = data[k:k + 4]
            target_input = target[k:k+4]
            output = model(data_input)
            loss = Dice_loss(output, target_input) + BCE_loss(torch.sigmoid(output), target_input)
            loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset), 100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(), time.time() - t, ), end="", )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset), 100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator), time.time() - t, ) )

    return np.mean(loss_accumulator)


@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure, phase):
    t = time.time()
    model.eval()
    perf_accumulator = []
    mIOU = []
    Dice = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        mIOU.append(Fmstric.binary_jaccard_index(torch.sigmoid(output), target>0.5).item())
        Dice.append(torchmetrics.functional.dice(torch.sigmoid(output), target>0.5).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\tDice: {:.6f}\tmIOU: {:.6f}\tDice: {:.6f}\tTime: {:.6f}".format(
                    phase, epoch, batch_idx + 1, len(test_loader), 100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator), np.mean(mIOU), np.mean(Dice), time.time() - t, ), end="", )
        else:
            print(
                "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\tDice: {:.6f}\tmIOU: {:.6f}\tDice: {:.6f}\tTime: {:.6f}".format(
                    phase,epoch, batch_idx + 1, len(test_loader), 100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator), np.mean(mIOU), np.mean(Dice), time.time() - t, ))

    return np.mean(perf_accumulator), np.std(perf_accumulator)



def train(args):
    ( device, train_dataloader, val_dataloader, test_dataloader,test_dataloader4vis, Dice_loss,
    BCE_loss, perf, model, optimizer, checkpoint, scheduler, loss_fun) = build(args)
    
    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    prev_best_test = checkpoint["val_measure_mean"]
    print("best val:", prev_best_test, "epoch:", checkpoint["epoch"])
    
    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, loss_fun["Dice_loss"], loss_fun["BCE_loss"]
            )
            val_measure_mean, val_measure_std = test(
                model, device, val_dataloader, epoch, perf,"Val"
            )
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs == "true":
            if args.type_lr == "LROnP":
                scheduler.step(val_measure_mean)
            else:
                scheduler.step()
        if prev_best_test == None or val_measure_mean > prev_best_test:
            print("Saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),
                    "loss": loss,
                    "val_measure_mean": val_measure_mean,
                    "val_measure_std": val_measure_std,
                },
                f"./Trained models/CTDCformer_epoch_backbonePvitB4_" + args.dataset + ".pt",
            )
            prev_best_test = val_measure_mean

class Args:
        def __init__(self,root, epochs, batch_size, dataset, mgpu, lrs_min,\
                    lrs, lr, type_lr, checkpoint_path, backbone, optim):
            self.root = root
            self.epochs = epochs
            self.batch_size = batch_size
            self.dataset = dataset
            self.mgpu = mgpu
            self.lrs_min = lrs_min
            self.lrs = lrs
            self.lr = lr
            self.type_lr = type_lr
            self.checkpoint_path = checkpoint_path
            self.backbone = backbone
            self.optim = optim
        
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
    checkpoint_path = None,
    backbone="PvtB3",
    optim="AdamW"
)

def main():

    train(args)


if __name__ == "__main__":
    main()
