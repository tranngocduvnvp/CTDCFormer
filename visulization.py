from lib import *
from train import model, test_dataloader4vis, device, perf


def postprocess_image(image):
    predicted_map = np.array(image.detach().cpu())
    predicted_map = np.squeeze(predicted_map)
    predicted_map = predicted_map > 0
    return predicted_map

def de_normal(image):
    image = image[0].permute(1,2,0).cpu().numpy()
    image = image*0.5+0.5
    return image

def saveImage(data, label, predict, grad_cam, path, is_save = True):
    plt.subplot(1,4,1)
    plt.imshow(data)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,4,2)
    plt.imshow(label, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,4,3)
    plt.imshow(predict, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,4,4)
    plt.imshow(grad_cam)
    plt.xticks([])
    plt.yticks([])
    if is_save == True:
        plt.savefig(path)


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()




def make_grad_Cam(image_tensor, mask_tensor, cam):
    rgb_img = de_normal(image_tensor)
    # print(rgb_img.shape)
    polyp_mask_float = postprocess_image(mask_tensor)
    # print(polyp_mask_float.shape)

    # print(model.attention1)
    polyp_category = 0
    # target_layers = [model.attention4]
    targets = [SemanticSegmentationTarget(polyp_category, polyp_mask_float)]
    # cam =  GradCAM(model=model,
    #             target_layers=target_layers,
    #             use_cuda=torch.cuda.is_available())

    grayscale_cam = cam(input_tensor=image_tensor,
                        targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    img_cam = Image.fromarray(cam_image)
    return rgb_img, polyp_mask_float, img_cam


def predict(data_name, cam):
    
    if not os.path.exists("./Predictions"):
        os.makedirs("./Predictions")
    if not os.path.exists("./Predictions/Trained on {}".format(data_name)):
        os.makedirs("./Predictions/Trained on {}".format(data_name))
    
    t = time.time()
    model.eval()
    perf_accumulator = []
    mIOU = []
    Dice = []
    for batch_idx, (data, target) in enumerate(test_dataloader4vis):
        data, target = data.to(device), target.to(device)
        output = model(data)
        rgb_img, polyp_mask_float, img_cam = make_grad_Cam(data, target, cam)
        perf_accumulator.append(perf(output, target).item())
        mIOU.append(Fmstric.binary_jaccard_index(torch.sigmoid(output), target>0.5).item())
        Dice.append(torchmetrics.functional.dice(torch.sigmoid(output), target>0.5).item())
        # input_image = de_normal(data)
        # labels = postprocess_image(target)
        predicted_map = postprocess_image(output)
        saveImage(rgb_img, polyp_mask_float, predicted_map, img_cam, "./Predictions/Trained on {}/dice_{}_{}.jpg".format(
                data_name, perf_accumulator[-1], batch_idx))
        # break
        if batch_idx + 1 < len(test_dataloader4vis):
            print(
                "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\tDice: {:.6f}\tmIOU: {:.6f}\tDice: {:.6f}\tTime: {:.6f}".format(
                    "Predict", 0, batch_idx + 1, len(test_dataloader4vis), 100.0 * (batch_idx + 1) / len(test_dataloader4vis),
                    np.mean(perf_accumulator), np.mean(mIOU), np.mean(Dice), time.time() - t, ), end="", )
        else:
            print(
                "\r{}  Epoch: {} [{}/{} ({:.1f}%)]\tDice: {:.6f}\tmIOU: {:.6f}\tDice: {:.6f}\tTime: {:.6f}".format(
                    "Predict",0, batch_idx + 1, len(test_dataloader4vis), 100.0 * (batch_idx + 1) / len(test_dataloader4vis),
                    np.mean(perf_accumulator), np.mean(mIOU), np.mean(Dice), time.time() - t, ))

    return np.mean(perf_accumulator), np.std(perf_accumulator)

if __name__ =="__main__":
    target_layers = [model.attention4]
    cam =  GradCAM(model=model,
                target_layers=target_layers,
                use_cuda=torch.cuda.is_available())
    predict(args.root.split("/")[-1], cam)