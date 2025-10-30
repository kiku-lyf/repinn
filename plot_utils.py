import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import matplotlib
import scipy
from scipy.interpolate import griddata
from util import *
from model_dict import get_model

# --- START OF MODIFICATIONS: Font and Size Configuration ---
# Use a backend that doesn't require a GUI
matplotlib.use('Agg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure Matplotlib for publication-quality plots
# This block sets all fonts to Times New Roman and controls the sizes.
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],

    # 默认文字大小（如图例、注释）
    'font.size': 10,

    # 子图标题和坐标轴标签
    'axes.titlesize': 14,  # 标题
    'axes.labelsize': 18,  # x,y 轴标签

    # 刻度标签
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,

    # 色条标签和刻度
    'legend.fontsize': 10,
    'figure.titlesize': 16,
})


# --- END OF MODIFICATIONS ---


def load_model_and_predict(model_path, model_name, device='cuda:0', x_range=[0, 2 * np.pi], t_range=[0, 1],
                           grid_size=101):
    """
    加载模型并进行预测

    Args:
        model_path: 模型文件路径
        model_name: 模型名称
        device: 设备
        x_range: x轴范围
        t_range: t轴范围
        grid_size: 网格大小

    Returns:
        pred: 预测结果
        x_star: x坐标
        t_star: t坐标
        XX: x网格
        TT: t网格
    """
    # 获取模型模块，并按训练脚本的分支方式实例化
    model_module = get_model(type('args', (), {'model': model_name})())
    if model_name == 'KAN':
        model = model_module.Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25,
                                   device=device).to(device)
    elif model_name == 'QRes':
        model = model_module.Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(device)
    elif model_name == 'PINNsFormer' or model_name == 'PINNsFormer_Enc_Only':
        model = model_module.Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(device)
    else:
        model = model_module.Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 注意：KAN 覆盖了 nn.Module.train 接口签名，调用 eval() 会触发其自定义 train(False)
    # 为避免报错，KAN 不调用 eval()；其他模型保持 eval()
    if model_name != 'KAN':
        model.eval()

    # 生成测试数据
    res_test, _, _, _, _ = get_data(x_range, t_range, grid_size, grid_size)
    x_star = np.linspace(x_range[0], x_range[1], grid_size)
    t_star = np.linspace(t_range[0], t_range[1], grid_size)
    TT, XX = np.meshgrid(t_star, x_star)

    # 转换为Tensor
    res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
    x_test, t_test = res_test[:, 0:1], res_test[:, 1:2]

    # 模型预测
    with torch.no_grad():
        pred = model(x_test, t_test)[:, 0:1]
        pred = pred.cpu().numpy().reshape(-1)

    return pred, x_star, t_star, XX, TT


def plot_solution_comparison(pred, exact, XX, TT, title_prefix="", save_path="result.pdf", exact_transpose=True):
    """
    绘制解的比较图（真实解、预测解、误差）

    Args:
        pred: 预测解
        exact: 真实解
        XX: x网格
        TT: t网格
        title_prefix: 标题前缀
        save_path: 保存路径
    """
    pred_grid = pred.reshape(XX.shape)

    # 创建画布
    fig = plt.figure(figsize=(18, 5))

    # 子图1：真实解
    plt.subplot(1, 3, 1)
    exact_grid = exact.T if exact_transpose else exact
    plt.pcolormesh(TT, XX, exact_grid, cmap='jet', shading='auto')
    cb = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(f'{title_prefix}Exact Solution')

    # 子图2：预测解
    plt.subplot(1, 3, 2)
    plt.pcolormesh(TT, XX, pred_grid.T, cmap='jet', shading='auto')
    cb = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(f'{title_prefix}Predicted Solution')

    # 子图3：绝对误差
    plt.subplot(1, 3, 3)
    plt.pcolormesh(TT, XX, np.abs(exact_grid - pred_grid.T), cmap='jet', shading='auto')
    cb = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(f'{title_prefix}Absolute Error')

    # 统一保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    plt.close()

    # 计算误差
    exact_flat = exact_grid.flatten()
    rl2 = np.linalg.norm(exact_flat - pred, 2) / np.linalg.norm(exact_flat, 2)
    rl1 = np.mean(np.abs(exact_flat - pred)) / np.mean(np.abs(exact_flat))

    print(f'Relative L1 error: {rl1:.4e}')
    print(f'Relative L2 error: {rl2:.4e}')

    return rl1, rl2


def plot_single_solution(pred, XX, TT, title="Solution", save_path="solution.pdf"):
    """
    绘制单个解

    Args:
        pred: 预测解
        XX: x网格
        TT: t网格
        title: 标题
        save_path: 保存路径
    """
    pred_grid = pred.reshape(XX.shape)

    # 创建画布
    fig = plt.figure(figsize=(8, 6))

    plt.pcolormesh(TT, XX, pred_grid.T, cmap='jet', shading='auto')
    cb = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    plt.close()


def plot_error_analysis(pred, exact, XX, TT, title_prefix="", save_path="error_analysis.pdf", exact_transpose=True):
    """
    绘制误差分析图

    Args:
        pred: 预测解
        exact: 真实解
        XX: x网格
        TT: t网格
        title_prefix: 标题前缀
        save_path: 保存路径
    """
    pred_grid = pred.reshape(XX.shape)
    exact_grid = exact.T if exact_transpose else exact
    error = np.abs(exact_grid - pred_grid.T)

    # 创建画布
    fig = plt.figure(figsize=(15, 5))

    # 子图1：绝对误差
    plt.subplot(1, 3, 1)
    plt.pcolormesh(TT, XX, error, cmap='jet', shading='auto')
    cb = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(f'{title_prefix}Absolute Error')

    # 子图2：相对误差
    plt.subplot(1, 3, 2)
    relative_error = error / (np.abs(exact_grid) + 1e-8)
    plt.pcolormesh(TT, XX, relative_error, cmap='jet', shading='auto')
    cb = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(f'{title_prefix}Relative Error')

    # 子图3：误差分布直方图
    plt.subplot(1, 3, 3)
    plt.hist(error.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title(f'{title_prefix}Error Distribution')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    plt.close()
