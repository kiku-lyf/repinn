#!/usr/bin/env python3
"""
对流方程点优化结果画图脚本
"""

import numpy as np
import torch
import argparse
import os
import scipy.io
from plot_utils import load_model_and_predict, plot_solution_comparison, plot_error_analysis


def load_convection_data(data_path='../../convection.mat'):
    """
    加载对流方程的真实解数据

    Args:
        data_path: 数据文件路径

    Returns:
        exact: 真实解数据
    """
    try:
        mat = scipy.io.loadmat(data_path)
        return mat['u']
    except FileNotFoundError:
        print(f"警告: 无法找到数据文件 {data_path}")
        print("将使用解析解代替")
        return None


def u_res(x, t):
    """对流方程的解析解"""
    return np.sin(x - 50 * t)


def main():
    parser = argparse.ArgumentParser('Plot Convection Point Optimization Results')
    parser.add_argument('--model', type=str, default='PINN',
                        choices=['PINN', 'QRes', 'FLS', 'KAN', 'PINNsFormer', 'PINNsFormer_Enc_Only'],
                        help='模型类型')
    parser.add_argument('--model_path', type=str, default='./results/convection_PINN_point.pt',
                        help='模型文件路径')
    parser.add_argument('--data_path', type=str, default='../../convection.mat',
                        help='真实解数据文件路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--output_dir', type=str, default='./plots', help='输出目录')
    parser.add_argument('--grid_size', type=int, default=101, help='网格大小')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置参数
    x_range = [0, 2 * np.pi]
    t_range = [0, 1]

    # 如果用户未修改默认路径且模型不是PINN，则自动切换到对应模型的默认权重文件
    if args.model != 'PINN' and args.model_path == './results/convection_PINN_point.pt':
        args.model_path = f'./results/convection_{args.model}_point.pt'

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

    # 获取真实解
    print("正在加载真实解数据...")
    exact_data = load_convection_data(args.data_path)

    if exact_data is not None:
        # 使用加载的真实解数据（保持原始轴顺序，后续在绘图函数中关闭转置）
        exact = exact_data
        print(f"已加载真实解数据，形状: {exact.shape}")
    else:
        # 使用解析解
        print("正在计算解析解...")
        exact = np.zeros_like(XX)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                exact[i, j] = u_res(XX[i, j], TT[i, j])

    # 绘制解的比较
    print("正在绘制解的比较图...")
    save_path = os.path.join(args.output_dir, f'convection_{args.model}_point_comparison.pdf')
    rl1, rl2 = plot_solution_comparison(
        pred=pred,
        exact=exact,
        XX=XX,
        TT=TT,
        title_prefix=f"Convection ({args.model}) - ",
        save_path=save_path,
        exact_transpose=False
    )

    # 绘制误差分析
    print("正在绘制误差分析图...")
    error_save_path = os.path.join(args.output_dir, f'convection_{args.model}_point_error.pdf')
    plot_error_analysis(
        pred=pred,
        exact=exact,
        XX=XX,
        TT=TT,
        title_prefix=f"Convection ({args.model}) - ",
        save_path=error_save_path,
        exact_transpose=False
    )

    print(f"图片已保存到: {args.output_dir}")
    print(f"相对L1误差: {rl1:.4e}")
    print(f"相对L2误差: {rl2:.4e}")


if __name__ == "__main__":
    main()
