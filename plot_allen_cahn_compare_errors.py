#!/usr/bin/env python3
"""
同时对比多个模型在Allen–Cahn方程上的误差热图
修复：
1. 添加批处理预测，避免 PINNsFormer OOM
2. 修正维度顺序问题
"""

import argparse
import os
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io

from model_dict import get_model
from util import make_time_sequence


def build_model(model_name: str, device: str):
    """构建模型"""
    module = get_model(type('args', (), {'model': model_name})())
    if model_name == 'KAN':
        model = module.Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0,
                             noise_scale_base=0.25, device=device).to(device)
    elif model_name == 'SetPINN':
        model = module.Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=3).to(device)
    elif model_name == 'QRes':
        model = module.Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(device)
    elif model_name == 'PINNsFormer' or model_name == 'PINNsFormer_Enc_Only':
        model = module.Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(device)
    else:
        model = module.Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
    return model


def predict_in_batches(model, x_test, t_test, batch_size=1000):
    """分批预测，避免 Transformer 类模型 OOM"""
    model.eval()
    n_samples = x_test.shape[0]
    predictions = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            x_batch = x_test[i:end_idx]
            t_batch = t_test[i:end_idx]

            pred_batch = model(x_batch, t_batch)[:, 0:1]
            predictions.append(pred_batch.cpu().numpy())

            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return np.concatenate(predictions, axis=0)


def config_matplotlib():
    """配置matplotlib样式"""
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
    })


def main():
    parser = argparse.ArgumentParser('Compare multiple models: Allen–Cahn errors')
    parser.add_argument('--models', nargs='+', required=False,
                        default=['PINN'],
                        help='模型名称列表，如: PINN KAN QRes PINNsFormer')
    parser.add_argument('--model_paths', nargs='*', default=None,
                        help='模型权重列表（与models一一对应）')
    parser.add_argument('--data_path', type=str, default='./AC.mat',
                        help='Allen–Cahn真实解数据文件路径')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./plots')
    parser.add_argument('--filename', type=str, default='allen_cahn_models_errors.pdf')
    parser.add_argument('--cols', type=int, default=3, help='每行显示的子图列数')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='PINNsFormer 预测时的批大小')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config_matplotlib()

    # 读取真实解数据
    print(f'加载真实解数据: {args.data_path}')
    mat = scipy.io.loadmat(args.data_path)
    Exact = mat['uu']  # shape: (nt, nx)
    t_star = mat['tt'].flatten()
    x_star = mat['x'].flatten()

    print(f'  Exact shape: {Exact.shape}')
    print(f'  x: [{x_star.min():.3f}, {x_star.max():.3f}], n={len(x_star)}')
    print(f'  t: [{t_star.min():.3f}, {t_star.max():.3f}], n={len(t_star)}')

    # 生成网格
    TT_ref, XX_ref = np.meshgrid(t_star, x_star)
    data = np.hstack([XX_ref.reshape(-1, 1), TT_ref.reshape(-1, 1)])

    print(f'  测试数据总点数: {data.shape[0]}')

    models = args.models
    if args.model_paths is None or len(args.model_paths) == 0:
        model_paths = [f'./results/allen_cahn_{m}_point.pt' for m in models]
    else:
        if len(args.model_paths) != len(models):
            raise ValueError('models 与 model_paths 数量不一致')
        model_paths = args.model_paths

    errors = []
    metrics = []

    # 逐模型预测
    for model_name, model_path in zip(models, model_paths):
        if not os.path.exists(model_path):
            print(f'警告: 模型权重不存在: {model_path}，跳过')
            continue

        try:
            print(f'\n加载模型: {model_name} | {model_path}')
            model = build_model(model_name, args.device)
            state = torch.load(model_path, map_location=args.device)
            model.load_state_dict(state)

            if model_name != 'KAN':
                model.eval()

            # 准备测试数据
            test_data = data.copy()

            # PINNsFormer 需要时间序列
            if model_name == 'PINNsFormer' or model_name == 'PINNsFormer_Enc_Only':
                print(f'  构造时间序列...')
                test_data = make_time_sequence(test_data, num_step=3, step=1e-4)

            res_test = torch.tensor(test_data, dtype=torch.float32,
                                    requires_grad=True).to(args.device)
            x_test, t_test = res_test[:, ..., 0:1], res_test[:, ..., 1:2]

            print(f'  预测中... (批大小={args.batch_size})')

            # 使用批处理预测
            if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only']:
                pred = predict_in_batches(model, x_test, t_test, args.batch_size)
            else:
                with torch.no_grad():
                    pred = model(x_test, t_test)[:, 0:1]
                    pred = pred.cpu().numpy()

            pred = pred.reshape(-1)
            pred_grid = pred.reshape(XX_ref.shape)

            # 误差计算
            Exact_flat = Exact.flatten()
            err_grid = np.abs(Exact - pred_grid)
            errors.append(err_grid)

            rl2 = np.linalg.norm(Exact_flat - pred, 2) / np.linalg.norm(Exact_flat, 2)
            rl1 = np.mean(np.abs(Exact_flat - pred)) / np.mean(np.abs(Exact_flat))
            metrics.append((model_name, rl1, rl2))

            print(f'  ✓ {model_name}: L1={rl1:.4e}, L2={rl2:.4e}')

            # 释放显存
            del model, state, res_test, x_test, t_test, pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f'  ✗ {model_name} 失败: {str(e)}')
            import traceback
            traceback.print_exc()
            continue

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
            im = ax.pcolormesh(TT_ref, XX_ref, err, cmap='jet',
                               shading='auto', vmin=0.0, vmax=vmax)
            ax.set_xlabel('$t$')
            ax.set_ylabel('$x$')
            ax.set_title(f'Allen–Cahn ({name}) | L1: {rl1:.2e}, L2: {rl2:.2e}')
            idx += 1

    # 统一色条
    fig.subplots_adjust(right=0.88, wspace=0.25, hspace=0.35)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    save_path = os.path.join(args.output_dir, args.filename)
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f'\n✅ 完成! 图像已保存: {save_path}')
    print('\n误差统计:')
    for name, rl1, rl2 in metrics:
        print(f'  {name}: L1={rl1:.4e}, L2={rl2:.4e}')


if __name__ == '__main__':
    main()
