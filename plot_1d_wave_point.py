#!/usr/bin/env python3
"""
1D波动方程点优化结果画图脚本
"""

import numpy as np
import torch
import argparse
import os
from plot_utils import load_model_and_predict, plot_solution_comparison, plot_error_analysis


def u_ana(x, t):
    """1D波动方程的解析解"""
    return np.sin(np.pi * x) * np.cos(2 * np.pi * t) + 0.5 * np.sin(3 * np.pi * x) * np.cos(6 * np.pi * t)


def main():
    parser = argparse.ArgumentParser('Plot 1D Wave Point Optimization Results')
    parser.add_argument('--model', type=str, default='PINN',
                        choices=['PINN', 'QRes', 'FLS', 'KAN', 'PINNsFormer', 'PINNsFormer_Enc_Only'],
                        help='模型类型')
    parser.add_argument('--model_path', type=str, default='./results/1dwave_PINN_point.pt',
                        help='模型文件路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--output_dir', type=str, default='./plots', help='输出目录')
    parser.add_argument('--grid_size', type=int, default=101, help='网格大小')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置参数
    x_range = [0, 1]
    t_range = [0, 1]

    print(f"正在加载模型: {args.model_path}")
    print(f"使用设备: {args.device}")

    # 加载模型并进行预测
    pred, x_star, t_star, XX, TT = load_model_and_predict(
        model_path=args.model_path,
        model_name=args.model,
        device=args.device,
        x_range=x_range,
        t_range=t_range,
        grid_size=args.grid_size
    )

    # 计算解析解
    print("正在计算解析解...")
    exact = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            exact[i, j] = u_ana(XX[i, j], TT[i, j])

    # 绘制解的比较
    print("正在绘制解的比较图...")
    save_path = os.path.join(args.output_dir, f'1d_wave_{args.model}_point_comparison.pdf')
    rl1, rl2 = plot_solution_comparison(
        pred=pred,
        exact=exact,
        XX=XX,
        TT=TT,
        title_prefix=f"1D Wave ({args.model}) - ",
        save_path=save_path,
        exact_transpose=False
    )

    # 绘制误差分析
    print("正在绘制误差分析图...")
    error_save_path = os.path.join(args.output_dir, f'1d_wave_{args.model}_point_error.pdf')
    plot_error_analysis(
        pred=pred,
        exact=exact,
        XX=XX,
        TT=TT,
        title_prefix=f"1D Wave ({args.model}) - ",
        save_path=error_save_path,
        exact_transpose=False
    )

    print(f"图片已保存到: {args.output_dir}")
    print(f"相对L1误差: {rl1:.4e}")
    print(f"相对L2误差: {rl2:.4e}")


if __name__ == "__main__":
    main()
