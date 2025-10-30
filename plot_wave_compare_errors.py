#!/usr/bin/env python3
"""
同时对比多个模型在1D波动方程上的误差热图

特性：
- 支持传入多个模型名与模型权重路径
- 自动计算解析解并与各模型预测对比
- 统一色标范围，便于横向比较
- 标题中显示相对L1/L2误差
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from plot_utils import load_model_and_predict


def u_ana(x, t):
    """1D波动方程解析解"""
    return np.sin(np.pi * x) * np.cos(2 * np.pi * t) + 0.5 * np.sin(3 * np.pi * x) * np.cos(6 * np.pi * t)


def compute_exact(XX, TT):
    exact = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            exact[i, j] = u_ana(XX[i, j], TT[i, j])
    return exact


def main():
    parser = argparse.ArgumentParser('Compare multiple models: 1D Wave errors')
    parser.add_argument('--models', nargs='+', required=False,
                        default=['PINN'],
                        help='模型名称列表，如: PINN KAN QRes')
    parser.add_argument('--model_paths', nargs='*', default=None,
                        help='模型权重列表（与models一一对应）。若不提供则按默认规则从./results推断')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--grid_size', type=int, default=101)
    parser.add_argument('--output_dir', type=str, default='./plots')
    parser.add_argument('--filename', type=str, default='wave_models_errors.pdf')
    parser.add_argument('--cols', type=int, default=3, help='每行显示的子图列数')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    x_range = [0, 1]
    t_range = [0, 1]

    models = args.models
    # 推断模型路径
    if args.model_paths is None or len(args.model_paths) == 0:
        model_paths = [f'./results/1dwave_{m}_point.pt' for m in models]
    else:
        if len(args.model_paths) != len(models):
            raise ValueError('models 与 model_paths 数量不一致')
        model_paths = args.model_paths

    preds = []
    errors = []
    metrics = []  # (rl1, rl2)

    XX_ref = None
    TT_ref = None
    exact_ref = None

    # 逐模型预测
    for model_name, model_path in zip(models, model_paths):
        if not os.path.exists(model_path):
            print(f'警告: 模型权重不存在: {model_path}，跳过该模型')
            continue

        print(f'加载模型: {model_name} | {model_path}')
        pred, x_star, t_star, XX, TT = load_model_and_predict(
            model_path=model_path,
            model_name=model_name,
            device=args.device,
            x_range=x_range,
            t_range=t_range,
            grid_size=args.grid_size
        )

        if XX_ref is None:
            XX_ref, TT_ref = XX, TT
            exact_ref = compute_exact(XX_ref, TT_ref)

        pred_grid = pred.reshape(XX.shape)
        pred_grid_T = pred_grid.T

        # 误差与指标（与单模型脚本保持一致：场展示用 *.T，误差使用 exact(x,t) - pred^T）
        err_grid = np.abs(exact_ref - pred_grid_T)
        errors.append(err_grid)

        exact_flat = exact_ref.flatten()
        pred_flat = pred_grid.flatten()
        rl2 = np.linalg.norm(exact_flat - pred_flat, 2) / np.linalg.norm(exact_flat, 2)
        rl1 = np.mean(np.abs(exact_flat - pred_flat)) / np.mean(np.abs(exact_flat))
        metrics.append((model_name, rl1, rl2))

    if len(errors) == 0:
        print('无可用模型，结束。')
        return

    # 统一色标范围
    vmax = max(np.max(err) for err in errors)

    # 画图
    n = len(errors)
    cols = max(1, args.cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx >= n:
                ax.axis('off')
                continue
            err = errors[idx]
            name, rl1, rl2 = metrics[idx]
            im = ax.pcolormesh(TT_ref, XX_ref, err, cmap='jet', shading='auto', vmin=0.0, vmax=vmax)
            ax.set_xlabel('$t$')
            ax.set_ylabel('$x$')
            ax.set_title(f'{name} | L1: {rl1:.2e}, L2: {rl2:.2e}')
            idx += 1

    # 统一色条
    fig.subplots_adjust(right=0.88, wspace=0.25, hspace=0.35)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    save_path = os.path.join(args.output_dir, args.filename)
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print('完成，多模型误差对比图已保存: ', save_path)
    for name, rl1, rl2 in metrics:
        print(f'- {name}: L1={rl1:.4e}, L2={rl2:.4e}')


if __name__ == '__main__':
    main()


