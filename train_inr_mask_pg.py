import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, TotalVariation
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from datasets import SliceDataset, ImageFittingPG
from transforms import INRDataTransformPG, image_to_kspace, kspace_to_image_abs

from fastmri.data.subsample import create_mask_for_mask_type
from inr.inr_model import SelfSirenKspaceMasking, SelfSirenResidualKspaceMasking

from loss import laplacian_edge, horizontal_total_variation, MixL1L2Loss


def seed_everything(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nmse_torch(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("The shape of y_true and y_pred must be the same.")

    # to numpy
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()

    # mean value
    y_mean = np.mean(y_true_np)

    # calculate NMSE
    numerator = np.sum((y_true_np - y_pred_np) ** 2)
    denominator = np.sum((y_true_np - y_mean) ** 2)

    nmse_score = numerator / denominator

    return nmse_score


def train_self_siren(net, optimizer, dataloader, total_steps=4000, steps_til_summary=50):
    coords, pos_encoding, d_pixels, pixels, d_image, gt_image, under_mask, remain_mask, select_mask = next(iter(dataloader))
    plt.imsave(inp_path / "input.png", d_image.detach().cpu().numpy().squeeze(), cmap="gray", vmax=d_image.max())
    np.save(npy_path / "input.npy", d_image.detach().cpu().numpy())
    plt.imsave(gt_path / "gt.png", gt_image.detach().cpu().numpy().squeeze(), cmap="gray", vmax=gt_image.max())
    np.save(gt_path / "gt.npy", gt_image.detach().cpu().numpy())
    coords, pos_encoding, d_pixels, pixels, d_image, gt_image, under_mask, remain_mask, select_mask = (coords.cuda(),
                                                                                                       pos_encoding.cuda(),
                                                                                                       d_pixels.cuda(),
                                                                                                       pixels.cuda(),
                                                                                                       d_image.cuda(),
                                                                                                       gt_image.cuda(),
                                                                                                       under_mask.cuda(),
                                                                                                       remain_mask.cuda(),
                                                                                                       select_mask.cuda())
    under_image = kspace_to_image_abs(image_to_kspace(gt_image) * under_mask)
    print(f"Input SSIM: {ssim(under_image, gt_image)}")
    input_ssim = float(ssim(under_image, gt_image))
    input_psnr = float(psnr(under_image, gt_image))
    input_mse = float(mse(under_image, gt_image))
    input_nmse = float(nmse_torch(gt_image, under_image))
    best_loss = np.inf
    best_ssim = -np.inf
    best_mse = np.inf
    best_nmse = np.inf
    best_psnr = -np.inf
    net.cuda()
    net.train()
    gt_kspace = image_to_kspace(gt_image)
    select_kspace = gt_kspace * select_mask
    with (tqdm(total=total_steps) as pbar):
        pbar.set_description('Training')
        for step in range(total_steps):
            model_input = torch.cat((pos_encoding, d_pixels), dim=-1)
            output_pixels, masked_img = net(model_input)
            masked_img = torch.clamp(masked_img, 0.0, 1.0)
            masked_pixels = masked_img.permute(1, 2, 0).view(-1, 1)
            output_img = output_pixels.view(1, image_size, image_size)
            output_img = torch.clamp(output_img, 0.0, 1.0)
            output_kspace = image_to_kspace(output_img)
            input_kspace = image_to_kspace(d_image)
            output_kr_kspace = output_kspace * (1 - under_mask) + gt_kspace * under_mask
            output_kr_img = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(output_kr_kspace)))
            output_kr_img = torch.clamp(output_kr_img, 0.0, 1.0)

            l_f_cons = (torch.abs(output_kspace * remain_mask - input_kspace * remain_mask) ** 2).mean() / (image_size**2)
            loss1 = ((d_pixels - masked_pixels) ** 2).mean()#.sum()
            loss2 = 1 - ssim(masked_img[None], d_image)
            l_i_cons = loss1 + loss2
            r_antiblur = laplacian_edge(output_img[None])
            r_tv = tv(output_img[None])
            l_f_inp = (torch.abs(output_kspace * select_mask - select_kspace) ** 2).mean() / (image_size**2)
            loss = l_f_inp + l_f_cons + l_i_cons + 0.001 * r_antiblur + 0.00001 * r_tv
            if not (step+1) % steps_til_summary:
                # ssim_value = ssim(output_pixels.view(1, 1, image_size, image_size), gt_image.view(1, 1, image_size, image_size))
                # psnr_value = psnr(output_pixels.view(1, 1, image_size, image_size), gt_image.view(1, 1, image_size, image_size))
                kr_ssim = ssim(output_kr_img.view(1, 1, image_size, image_size), gt_image.view(1, 1, image_size, image_size))
                kr_psnr = psnr(output_kr_img.view(1, 1, image_size, image_size), gt_image.view(1, 1, image_size, image_size))
                kr_mse = mse(output_kr_img.view(1, 1, image_size, image_size), gt_image.view(1, 1, image_size, image_size))
                kr_nmse = nmse_torch(gt_image.view(1, 1, image_size, image_size), output_kr_img.view(1, 1, image_size, image_size))
                # mse_value = mse(output_pixels.view(1, 1, image_size, image_size), gt_image.view(1, 1, image_size, image_size))
                # nmse_value = nmse_torch(gt_image.view(1, 1, image_size, image_size), output_pixels.view(1, 1, image_size, image_size))
                if loss.item() < best_loss:
                    checkpoint = {
                        'epoch': step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ssim': kr_ssim,
                        'psnr': kr_psnr,
                        'mse': kr_mse,
                        'nmse': kr_nmse,
                    }
                    # best_img = output_img.detach().cpu().numpy().squeeze()
                    best_kr_img = output_kr_img.detach().cpu().numpy().squeeze()
                    torch.save(checkpoint, ckpt_path / f"model_best.ckpt")
                    np.save(npy_path / f"best.npy", best_kr_img)
                    plt.imsave(fig_path / f"best.png", best_kr_img, cmap="gray")
                    best_ssim = float(kr_ssim)
                    best_mse = float(kr_mse)
                    best_nmse = float(kr_nmse)
                    best_psnr = float(kr_psnr)
                    best_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
    data = {"Slice_Num": slice_num, "Input SSIM": input_ssim, "Input PSNR": input_psnr, "Input MSE": input_mse,
            "Input NMSE": input_nmse, "Best SSIM": best_ssim, "Best PSNR": best_psnr, "Best MSE": best_mse}
    if csv_path.exists():
        # if file exists
        df = pd.DataFrame([data])
        df.to_csv(csv_path, mode='a', header=False, index=False, float_format='%.6f')
    else:
        # if file does not exist
        df = pd.DataFrame([data])
        df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Best SSIM: {best_ssim}")
    print(f"Best PSNR: {best_psnr}")
    print(f"Best MSE: {best_mse}")
    print(f"Best NMSE: {best_nmse}")

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="path_to_data", help='root path of dataset')
parser.add_argument('--protocol', type=str, default='PD')
parser.add_argument('--af', type=int, default=4, choices=[4, 6, 8])
parser.add_argument('--exp_type', type=str, default="ALL")
parser.add_argument('--slice_id', type=int, default=65)
parser.add_argument('--parent_folder', type=str, default="fastmri")
parser.add_argument('--results_csv_file', type=str, default="results.csv")
args = parser.parse_args()

root = Path(args.data_path)
seed_everything(42)
csv_path = "/mnt/data1/datasx/pd_singlecoil_crop.csv"
df = pd.read_csv(csv_path)
af = args.af
protocol = args.protocol
path_column = df['Path']
mask_func = create_mask_for_mask_type("equispaced_fraction", [0.08], [af])

data_transform = INRDataTransformPG(challenge='singlecoil', mask_func=mask_func)
dataset = SliceDataset(root, path_column, data_transform, True, f"fastmri_{protocol.lower()}_single_dataset_cache.pkl")

slice_num = args.slice_id
nums_img = len(dataset)
sample = dataset.__getitem__(slice_num)
image_size = sample.image.shape[-1]
dataset_for_siren = ImageFittingPG(image_size, sample, pos_dim=4)
dataloader = DataLoader(dataset_for_siren, batch_size=1, pin_memory=True, num_workers=0)

parent_folder = args.parent_folder
exp_type = args.exp_type

ckpt_path = Path("./ckpt/") / parent_folder / exp_type / (protocol + "_" + str(slice_num))
fig_path = Path("./figs/") / parent_folder / exp_type / (protocol + "_" + str(slice_num))
npy_path = Path("./npys/") / parent_folder / exp_type / (protocol + "_" + str(slice_num))
inp_path = Path("./inps/") / parent_folder / exp_type / (protocol + "_" + str(slice_num))
gt_path = Path("./gts/") / parent_folder / exp_type / (protocol + "_" + str(slice_num))
csv_path = Path("./csv_results") / parent_folder / args.results_csv_file


if not ckpt_path.exists():
    ckpt_path.mkdir(parents=True)
if not fig_path.exists():
    fig_path.mkdir(parents=True)
if not npy_path.exists():
    npy_path.mkdir(parents=True)
if not inp_path.exists():
    inp_path.mkdir(parents=True)
if not gt_path.exists():
    gt_path.mkdir(parents=True)
if not csv_path.parent.exists():
    csv_path.parent.mkdir(parents=True)

ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
mse = nn.MSELoss()
tv = TotalVariation().cuda()
img_siren = SelfSirenKspaceMasking(in_features=19, out_features=1, mask=sample.remain_mask, hidden_layers=7, hidden_features=256)
total_steps = 2500
steps_til_summary = 10
optim = torch.optim.Adam(lr=1e-4, params=filter(lambda p: p.requires_grad, img_siren.parameters()))
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=300, gamma=0.5)
train_self_siren(img_siren, optim, dataloader, args, total_steps=total_steps, steps_til_summary=steps_til_summary)
