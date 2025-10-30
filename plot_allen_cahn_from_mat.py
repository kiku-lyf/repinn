#!/usr/bin/env python3
"""
Allen–Cahn 方程（从 AC.mat 读取真值）画图脚本

数据文件字段约定：
- uu:  真值场，形状 (len(x), len(t)) 或 (len(t), len(x))，本脚本按示例使用 uu 与转置匹配
- tt:  时间向量 (len(t),)
- x:   空间向量 (len(x),)

绘图与误差计算严格参考用户给定代码的网格与转置方式，避免 x/t 轴颠倒。
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io

from model_dict import get_model


def build_model(model_name: str, device: str):
    module = get_model(type('args', (), {'model': model_name})())
    if model_name == 'KAN':
        model = module.Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25, device=device).to(device)
    elif model_name == 'QRes':
        model = module.Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(device)
    elif model_name == 'PINNsFormer' or model_name == 'PINNsFormer_Enc_Only':
        model = module.Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(device)
    else:
        model = module.Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
    return model


def config_matplotlib():
    import matplotlib
    matplotlib.rcParams.update({
        'font.family':        'serif',
        'font.serif':         ['Times New Roman'],
        'font.size':          10,
        'axes.titlesize':     14,
        'axes.labelsize':     18,
        'xtick.labelsize':    12,
        'ytick.labelsize':    12,
        'legend.fontsize':    10,
        'figure.titlesize':   16,
    })


def main():
    parser = argparse.ArgumentParser('Allen–Cahn plotting from mat file')
    parser.add_argument('--model', type=str, default='PINN',
                        choices=['PINN', 'QRes', 'FLS', 'KAN', 'PINNsFormer', 'PINNsFormer_Enc_Only'])
    parser.add_argument('--model_path', type=str, default='./results/allen_cahn_PINN_point.pt')
    parser.add_argument('--data_path', type=str, default='./AC.mat')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output', type=str, default='./plots/allen_cahn_from_mat.pdf')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    config_matplotlib()

    # 1) 读入真值数据
    mat = scipy.io.loadmat(args.data_path)
    Exact = mat['uu']
    t_star = mat['tt'].flatten()
    x_star = mat['x'].flatten()

    # 采用与项目其它脚本一致的网格顺序：先 t 再 x
    # 生成形状均为 (len(x), len(t)) 的网格
    TT, XX = np.meshgrid(t_star, x_star)

    # 2) 构造测试点并预测
    data = np.hstack([XX.reshape(-1, 1), TT.reshape(-1, 1)])
    res_test = torch.tensor(data, dtype=torch.float32, requires_grad=True).to(args.device)
    x_test, t_test = res_test[:, 0:1], res_test[:, 1:2]

    model = build_model(args.model, args.device)
    state = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state)
    if args.model != 'KAN':
        model.eval()

    with torch.no_grad():
        pred = model(x_test, t_test)[:, 0:1]
        pred = pred.cpu().numpy().reshape(-1)

    # 3) 误差：使用与网格一致的取向（Exact 与 XX/TT 同形状）
    Exact_flat = Exact.flatten()
    rl2 = np.linalg.norm(Exact_flat - pred, 2) / np.linalg.norm(Exact_flat, 2)
    rl1 = np.mean(np.abs(Exact_flat - pred)) / np.mean(np.abs(Exact_flat))
    print(f'Relative L1 error: {rl1:.4e}')
    print(f'Relative L2 error: {rl2:.4e}')

    # 4) 绘图
    pred_grid = pred.reshape(XX.shape)

    fig = plt.figure(figsize=(18, 5))

    # 子图1：真实解（不转置，确保与网格形状一致）
    plt.subplot(1, 3, 1)
    plt.pcolormesh(TT, XX, Exact, cmap='jet', shading='auto')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Allen–Cahn - Exact Solution')

    # 子图2：预测解（不转置，保持与网格形状一致）
    plt.subplot(1, 3, 2)
    plt.pcolormesh(TT, XX, pred_grid, cmap='jet', shading='auto')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Allen–Cahn - Predicted Solution')

    # 子图3：绝对误差（不转置）
    plt.subplot(1, 3, 3)
    plt.pcolormesh(TT, XX, np.abs(Exact - pred_grid), cmap='jet', shading='auto')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Allen–Cahn - Absolute Error')

    plt.tight_layout()
    plt.savefig(args.output, dpi=600, format='pdf', bbox_inches='tight')
    plt.close()
    print('Saved to:', args.output)


if __name__ == '__main__':
    main()


