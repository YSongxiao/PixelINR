import torch
import numpy as np
import matplotlib.pyplot as plt


def random_select_mask_torch(mask, percentage, h1):
    H, W = mask.shape
    selected_mask = torch.zeros_like(mask)
    remaining_mask = torch.zeros_like(mask)

    # 计算中心的起止点
    start_h = (H - h1) // 2
    end_h = start_h + h1
    start_w = (W - h1) // 2
    end_w = start_w + h1
    print(start_h, end_h)
    print(start_w, end_w)
    # 复制原mask中除了中心外的部分
    active_area = mask.clone()
    active_area[start_h:end_h, start_w:end_w] = 0

    # 在active_area中找到为1的位置
    active_indices = (active_area == 1).nonzero(as_tuple=True)
    num_active_pixels = active_indices[0].size(0)
    num_select = int(percentage * num_active_pixels)  # 直接使用percentage计算需要选择的数量

    # 随机选择指定数量的点
    if num_select > 0:
        selected_indices = torch.randperm(num_active_pixels)[:num_select]
        selected_mask[active_indices[0][selected_indices], active_indices[1][selected_indices]] = 1

    # 更新剩余的mask
    remaining_mask = mask.clone()
    remaining_mask[selected_mask == 1] = 0

    return selected_mask, remaining_mask


def random_select_mask_torch_full(mask, percentage):
    H, W = mask.shape
    selected_mask = torch.zeros_like(mask)
    remaining_mask = torch.zeros_like(mask)

    # 复制原mask中除了中心外的部分
    active_area = mask.clone()
    # active_area[start_h:end_h, start_w:end_w] = 0

    # 在active_area中找到为1的位置
    active_indices = (active_area == 1).nonzero(as_tuple=True)
    num_active_pixels = active_indices[0].size(0)
    num_select = int(percentage * num_active_pixels)  # 直接使用percentage计算需要选择的数量

    # 随机选择指定数量的点
    if num_select > 0:
        selected_indices = torch.randperm(num_active_pixels)[:num_select]
        selected_mask[active_indices[0][selected_indices], active_indices[1][selected_indices]] = 1

    # 更新剩余的mask
    remaining_mask = mask.clone()
    remaining_mask[selected_mask == 1] = 0

    return selected_mask, remaining_mask

# 示例使用
# mask = torch.tensor([[1, 1, 1, 0, 0],
#                      [1, 1, 1, 1, 0],
#                      [1, 1, 1, 1, 1],
#                      [0, 1, 1, 1, 1],
#                      [0, 0, 1, 1, 1]], dtype=torch.int32)
#
# selected_mask, remaining_mask = random_select_mask_torch(mask, 0.5, 1)
#
# print("Selected Mask:\n", selected_mask)
# print("Remaining Mask:\n", remaining_mask)
mask_ixi = np.load("./mask/mask_4x_equispaced_ixi.npy")
mask = np.load("./mask/mask_4x_equispaced.npy")
mask_ixi_tch = torch.tensor(mask_ixi)
mask_tch = torch.tensor(mask)
r = 0.99
# IXI 21x21 FastMRI 24x24
selected_mask, remaining_mask = random_select_mask_torch(mask_tch, r, 24)
selected_mask_ixi, remaining_mask_ixi = random_select_mask_torch(mask_ixi_tch, r, 21)
np.save("./mask/selected_mask_4x_ixi_0990.npy", selected_mask_ixi.numpy())
np.save("./mask/remained_mask_4x_ixi_0990.npy", remaining_mask_ixi.numpy())
np.save("./mask/selected_mask_4x_0990.npy", selected_mask.numpy())
np.save("./mask/remained_mask_4x_0990.npy", remaining_mask.numpy())
# plt.imsave("./mask/selected_mask_4x_ixi_0125.png", selected_mask, cmap="gray", vmax=1, vmin=0)
plt.imshow(selected_mask, cmap="gray")
plt.show()
plt.imshow(remaining_mask, cmap="gray")
plt.show()
# plt.imsave("./mask/remained_mask_4x_ixi_0125.png", remaining_mask, cmap="gray", vmax=1, vmin=0)
# plt.show()
