from lib import *
from train import model, device, test_dataloader4vis
from visulization import de_normal, postprocess_image

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

def showAll(image_rgb, label, cam, path='', is_save = False):
    plt.subplot(1,3,1)
    plt.imshow(image_rgb)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,2)
    plt.imshow(label, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,3)
    plt.imshow(cam)
    plt.xticks([])
    plt.yticks([])
    if is_save == True:
        plt.savefig(path)

if __name__ == "__main__":

    target_layers = [model.rb5]
    cam =  GradCAM(model=model,
                target_layers=target_layers,
                use_cuda=torch.cuda.is_available())
    image_tensor, mask_tensor = next(iter(test_dataloader4vis))
    rgb_img, polyp_mask_float, img = make_grad_Cam(image_tensor, mask_tensor, cam)
    showAll(rgb_img, polyp_mask_float, img)
    plt.show()
