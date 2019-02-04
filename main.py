import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
import cv2
from skimage.transform import resize
from imagenet_classes import class_names



def alpha_prior(x, alpha=2.):
    # the view function turn the 3/4 dimension values to one dimension
    # ** just do the same operation for each element of the vector
    return torch.abs(x.view(-1)**alpha).sum()
    # equal to: return torch.abs(x.pow(alpha)).sum()

# claculate the difference from the input image and the inverted image
def perceptual_loss(input, target):
    return torch.div(alpha_prior(input - target, alpha=2.), alpha_prior(target, alpha=2.))


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Clip(object):
    def __init__(self):
        return

    def __call__(self, tensor):
        t = tensor.clone()
        t[t>1] = 1
        t[t<0] = 0
        return t


#function to decay the learning rate
def decay_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor 


def get_pytorch_module(net, blob):
    modules = blob.split('.')
    if len(modules) == 1:
        return net._modules.get(blob)
    else:
        curr_m = net
        for m in modules:
            curr_m = curr_m._modules.get(m)
        return curr_m

def saveMask(mask):
    mask1 = mask.cpu().data.numpy()[0]
    # tranpose the image to BGR format
    mask1 = np.transpose(mask1, (1, 2, 0))
    # normalize the mask
    mask1 = (mask1 - np.min(mask1)) / np.max(mask1)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask1), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cv2.imwrite("heatmap.png", np.uint8(255 * heatmap))

def invert(image, network, layer, epochs, cuda, target_label):

    learning_rate = 1e-2
    size=224
    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Scale(size=size),  # Resize the input PIL Image to the given size.
        transforms.CenterCrop(size=size), # Crops the given PIL Image at the center.
        transforms.ToTensor(),   # from PIL.Image or numpy.ndarray, to values [0,1] 
        transforms.Normalize(mu, sigma), # channel = (channel-mean)/std
    ])

    detransform = transforms.Compose([
        Denormalize(mu, sigma),
        Clip(),
        transforms.ToPILImage(),
    ])


    model = models.__dict__[network](pretrained=True)
    model.eval()
    if cuda:
        model.cuda()

    # transform the input (PIL image) to tensors with required dimension
    # unsqueeze, add one demision to the image
    img_ = transform(Image.open(image)).unsqueeze(0)

    # process the blurred image
    image_blur_input = Image.open(image)
    blurred_img = image_blur_input.filter(ImageFilter.GaussianBlur(radius=11))
    image_blur = transform(blurred_img).unsqueeze(0)
    input_var_blur = Variable(image_blur.cuda() if cuda else image_blur)

    upsample = torch.nn.UpsamplingBilinear2d(size=(size, size))

    activations = []
    def hook_acts(module, input, output):
        activations.append(output)

    # get the model activation value
    def get_acts(model, input):
        # delete values in the list
        del activations[:]
        # will call the hook, and calculate the intermediate value of CNN
        _ = model(input)
        # if the length of the activations is not one, will raise error
        assert(len(activations) == 1)
        # return the activation value
        return activations[0]

    # Registers a forward hook on the module.
    # The hook will be called every time after forward() has computed an output.
    _ = get_pytorch_module(model, layer).register_forward_hook(hook_acts)

    # define the input image
    input_var = Variable(img_.cuda() if cuda else img_)
    logits = model(input_var)
    target = torch.nn.Softmax()(logits)
    y = np.argsort(target.cpu().data.numpy())
    category_highest = y[0][-1]
    category = y[0][-target_label]
    print "Category with highest probability", category
    print class_names[category], target.cpu().data.numpy()[0][category]
    target_logits = logits[0][category_highest]


    # add detach, so that we will not calculate gradient for this activation value
    ref_acts = get_acts(model, input_var).detach()
    ref_acts_size = ref_acts.size()[1]
    print "The size of the targeted intermediate layer:", ref_acts.size()


    w_init = torch.div(torch.ones(1, ref_acts_size), 10)
    w = Variable(w_init.cuda() if cuda else w_init, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=learning_rate)

    # the optimization process
    for i in range(epochs):
        upsampled_mask = w[0, ref_acts_size - 1] * ref_acts[0, ref_acts_size - 1, :, :]
        for j in range(ref_acts_size - 1):
            upsampled_mask += w[0, j] * ref_acts[0, j, :, :]

        upsampled_mask.data.unsqueeze_(0)
        upsampled_mask.data.unsqueeze_(0)
        upsampled_mask = upsample(upsampled_mask)

        # using normalization lead to better result
        upsampled_mask = (upsampled_mask - upsampled_mask.min()) / upsampled_mask.max()

        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))

        # Use the mask to perturbated the input image.
        x_ = input_var.mul(upsampled_mask) + input_var_blur.mul(1 - upsampled_mask)
        x_deleted = input_var.mul(1 - upsampled_mask) + input_var_blur.mul(upsampled_mask)

        # the Forward propagation, get the activation value for the inverted image
        acts = get_acts(model, x_)
        # data loss
        loss_term = perceptual_loss(acts, ref_acts) 
        #  norm
        norm_term = alpha_prior(w, 2)    
        # highlight loss
        logits_highlight = -1 * torch.nn.Softmax()(model(x_))[0][category]
        # suppressing loss
        logits_supress = torch.nn.Softmax()(model(x_deleted))[0][category]
        if i < 11:
            tot_loss = loss_term + 5 * norm_term 

        else:
            tot_loss = logits_highlight + 1 * logits_supress + 2 * norm_term
        # the result is a list, contains batch_size values, since the batch size is 1, we need [0]
        if (i+1) % 1 == 0:
            print('Epoch %d: \tTot Loss: %f' % (i+1, tot_loss.data.cpu().numpy()[0]))

        # zero gradients, perform a back propogation pass, and update the gradients
        optimizer.zero_grad()
        tot_loss.backward(retain_graph=True)
        optimizer.step()
        w.data.clamp_(0,100000)

        # define the learning rate decay
        if (i+1) % 10 == 0:
            decay_lr(optimizer, 0.5)

    upsampled_mask = w[0, ref_acts_size - 1] * ref_acts[0, ref_acts_size - 1, :, :]
    for j in range(ref_acts_size - 1):
        upsampled_mask += w[0, j] * ref_acts[0, j, :, :]
    upsampled_mask.data.unsqueeze_(0)
    upsampled_mask.data.unsqueeze_(0)
    upsampled_mask = upsample(upsampled_mask)
    mask_copy = upsampled_mask.cpu().data.numpy()[0,0]
    saveMask(upsampled_mask)


    # the following code is for visualization
    upsampled_mask = (upsampled_mask-upsampled_mask.min())/upsampled_mask.max()
    # The single channel mask is used with an RGB image,
    # so the mask is duplicated to have 3 channel,
    upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))
    # Use the mask to perturbated the input image.
    x_ = input_var.mul(upsampled_mask) + input_var_blur.mul(1 - upsampled_mask)
    # save image using PIL, x_[0] has the same resolution with image
    invert_pil = detransform(x_[0].data.cpu())
    invert_pil.save("inverted.jpg")
    heatmap = Image.open("heatmap.png")
    # show the final result
    f, ax = plt.subplots(1, 3)
    # transform back to PIL image
    ax[0].imshow(detransform(img_[0]))
    ax[1].imshow(invert_pil)
    ax[2].imshow(heatmap)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.show()


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--image', type=str, default='ILSVRC2012_val_00000187.JPEG')
        parser.add_argument('--network', type=str, default='vgg19')
        parser.add_argument('--layer', type=str, default='features.36')
        parser.add_argument('--epochs', type=int, default=80)
        parser.add_argument('--gpu', type=int, nargs='*', default=None)
        parser.add_argument('--label', type=int, default=1)

        args = parser.parse_args()

        gpu = args.gpu
        cuda = True if gpu is not None else False
        use_mult_gpu = isinstance(gpu, list)
        if cuda:
            if use_mult_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
        print(torch.cuda.device_count(), use_mult_gpu, cuda)

        invert(image=args.image, network=args.network, layer=args.layer, 
                epochs=args.epochs, cuda=cuda, target_label=args.label)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


