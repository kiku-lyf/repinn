#!/usr/bin/env python3
"""
同时对比多个模型在Wave方程上的误差热图
每个子图独立颜色条，与训练代码保持完全一致的数据处理方式
"""

import argparse
import os
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_dict import get_model
from util import get_data, make_time_sequence

# ============ 字体和样式配置 ============
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})


def build_model(model_name: str, device: str):
    """构建模型"""
    module = get_model(type('args', (), {'model': model_name})())
    if model_name == 'KAN':
        model = module.Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0,
                             noise_scale_base=0.25, device=device).to(device)
    elif model_name == 'QRes':
        model = module.Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(device)
    elif model_name == 'SetPINN':
        model = module.Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=3).to(device)
    elif model_name == 'PINNsFormer' or model_name == 'PINNsFormer_Enc_Only':
        model = module.Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(device)
    else:
        model = module.Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
    return model


def u_exact(x, t):
    """Wave方程的解析解"""
    return (np.sin(np.pi * x) * np.cos(2 * np.pi * t) +
            0.5 * np.sin(3 * np.pi * x) * np.cos(6 * np.pi * t))


def main():
    parser = argparse.ArgumentParser('Compare multiple models: Wave equation errors')
    parser.add_argument('--models', nargs='+', required=False,
                        default=['PINN', 'KAN', 'QRes'],
                        help='模型名称列表，如: PINN KAN QRes PINNsFormer')
    parser.add_argument('--model_paths', nargs='*', default=None,
                        help='模型权重列表（与models一一对应）')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./plots')
    parser.add_argument('--filename', type=str, default='wave_models_errors.pdf')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 使用与训练完全相同的方式生成测试数据
    print('生成测试数据（与训练代码一致）')
    res_test, _, _, _, _ = get_data([0, 1], [0, 1], 101, 101)

    # 计算精确解
    u_true = u_exact(res_test[:, 0], res_test[:, 1]).reshape(101, 101)
    print(f'  精确解形状: {u_true.shape}')

    # 创建网格坐标（用于pcolormesh）
    x_grid = np.linspace(0, 1, 101)
    t_grid = np.linspace(0, 1, 101)
    XX, TT = np.meshgrid(x_grid, t_grid)

    models = args.models
    # 推断模型路径
    if args.model_paths is None or len(args.model_paths) == 0:
        model_paths = [f'./results/1dwave_{m}_point.pt' for m in models]
    else:
        if len(args.model_paths) != len(models):
            raise ValueError('models 与 model_paths 数量不一致')
        model_paths = args.model_paths

    errors = []
    metrics = []

    # 逐模型预测
    for model_name, model_path in zip(models, model_paths):
        if not os.path.exists(model_path):
            print(f'警告: 模型权重不存在: {model_path}，跳过该模型')
            continue

        try:
            print(f'\n加载模型: {model_name} | {model_path}')
            model = build_model(model_name, args.device)

            print(f'  加载权重...')
            state = torch.load(model_path, map_location=args.device)
            model.load_state_dict(state)
            if model_name != 'KAN':
                model.eval()

            print(f'  准备测试数据...')
            # 与训练代码完全一致的数据处理
            test_data = res_test.copy()
            if model_name == 'PINNsFormer' or model_name == 'PINNsFormer_Enc_Only':
                test_data = make_time_sequence(test_data, num_step=5, step=1e-4)

            test_tensor = torch.tensor(test_data, dtype=torch.float32, requires_grad=True).to(args.device)
            x_test, t_test = test_tensor[:, ..., 0:1], test_tensor[:, ..., 1:2]

            print(f'  进行预测...')
            with torch.no_grad():
                pred = model(x_test, t_test)[:, 0:1]
                pred = pred.cpu().detach().numpy()

            # 与训练代码一致的 reshape
            pred_grid = pred.reshape(101, 101)

            # 误差计算
            err_grid = np.abs(u_true - pred_grid)
            errors.append(err_grid)

            # 计算指标（与训练代码一致）
            rl1 = np.sum(np.abs(u_true - pred_grid)) / np.sum(np.abs(u_true))
            rl2 = np.sqrt(np.sum((u_true - pred_grid) ** 2) / np.sum(u_true ** 2))
            metrics.append((model_name, rl1, rl2))

            print(f'  ✓ {model_name} 完成: L1={rl1:.4e}, L2={rl2:.4e}')

            # 释放内存
            del model, state, pred, pred_grid, test_tensor
            torch.cuda.empty_cache() if args.device.startswith('cuda') else None

        except Exception as e:
            print(f'  ✗ {model_name} 处理失败: {str(e)}')
            import traceback
            traceback.print_exc()
            continue

    if len(errors) == 0:
        print('无可用模型，结束。')
        return

    # ============ 绘图：每个子图独立颜色条 ============
    n_models = len(errors)
    n_cols = min(3, n_models)  # 每行最多3列
    n_rows = int(np.ceil(n_models / 3))  # 计算需要的行数

    # 使用 subplots 创建子图数组
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # 处理单个子图的情况
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    print(f'\n布局信息: {n_models}个模型，排列为 {n_rows} 行 × {n_cols} 列')

    # 逐模型绘制误差热图（每个子图独立颜色条）
    for idx, (err_grid, (model_name, rl1, rl2)) in enumerate(zip(errors, metrics)):
        ax = axes[idx]

        # ✅ 删除 vmax 参数，让每个子图自动缩放
        mesh = ax.pcolormesh(TT, XX, err_grid, cmap='jet',
                             shading='auto', vmin=0.0)

        # ✅ 为每个子图创建独立的颜色条
        cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

        # 使用数学文本格式的斜体变量
        ax.set_xlabel('$t$', fontsize=14)
        ax.set_ylabel('$x$', fontsize=14)

        # 标题只显示模型名称
        ax.set_title(f'{model_name}', fontsize=14, pad=10)
        ax.tick_params(labelsize=11)

    # 隐藏多余的子图
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, args.filename)
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f'\n✅ 完成! 误差对比图已保存: {save_path}')
    print('\n📊 误差统计 (按 L2 排序):')
    for name, rl1, rl2 in sorted(metrics, key=lambda x: x[2]):
        print(f'  {name:20s} L1={rl1:.4e}  L2={rl2:.4e}')


if __name__ == '__main__':
    main()
