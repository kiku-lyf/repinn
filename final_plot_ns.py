#!/usr/bin/env python3
"""
同时对比多个模型在NS方程上的压力场误差热图
每个子图独立颜色条，与 Burgers、Allen-Cahn 和 KdV 代码保持完全一致
"""

import argparse
import os
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io

from models1 import PINN, QRes, FLS, KAN, PINNsFormer, PINNsFormer_Enc_Only, SetPINN
from util import make_time_sequence

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


def get_model(args):
    model_dict = {
        'PINN': PINN,
        'QRes': QRes,
        'FLS': FLS,
        'KAN': KAN,
        'PINNsFormer': PINNsFormer,
        'PINNsFormer_Enc_Only': PINNsFormer_Enc_Only,
        'SetPINN': SetPINN,
    }
    return model_dict[args.model]


def build_model(model_name: str, device: str):
    """构建模型（统一接口）"""
    module = get_model(type('args', (), {'model': model_name})())

    if model_name == 'KAN':
        model = module.Model(width=[3, 5, 5, 2], grid=5, k=3, grid_eps=1.0,
                             noise_scale_base=0.25, device=device).to(device)
    elif model_name == 'SetPINN':
        model = module.Model(in_dim=3, hidden_dim=32, out_dim=2, num_layer=3).to(device)
    elif model_name == 'QRes':
        model = module.Model(in_dim=3, hidden_dim=256, out_dim=2, num_layer=4).to(device)
    elif model_name == 'PINNsFormer':
        model = module.Model(d_out=2, d_hidden=64, d_model=32, N=1, heads=2).to(device)
    elif model_name == 'PINNsFormer_Enc_Only':
        model = module.Model(d_out=2, d_hidden=512, d_model=32, N=1, heads=2).to(device)
    else:
        # PINN, FLS 等默认模型
        model = module.Model(in_dim=3, hidden_dim=512, out_dim=2, num_layer=4).to(device)

    return model


def predict_in_batches(model, model_name, x_test, y_test, t_test, batch_size=1000):
    """分批预测，避免 OOM"""
    model.eval()
    n_samples = len(x_test)
    predictions = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            x_batch = x_test[i:end_idx]
            y_batch = y_test[i:end_idx]
            t_batch = t_test[i:end_idx]

            # PINNsFormer 需要梯度
            if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only']:
                x_batch.requires_grad_(True)
                y_batch.requires_grad_(True)
                t_batch.requires_grad_(True)

                psi_and_p = model(x_batch, y_batch, t_batch)  # (batch, seq_len, 2)
                p_pred_batch = psi_and_p[:, 0, 1]  # 第一个时间步的压力
            else:
                psi_and_p = model(x_batch, y_batch, t_batch)  # (batch, 2)
                p_pred_batch = psi_and_p[:, 1]  # 压力

            predictions.append(p_pred_batch.cpu().numpy())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return np.concatenate(predictions, axis=0)


def load_ns_data(data_path):
    """加载 NS 方程数据"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'数据文件不存在: {data_path}')

    mat = scipy.io.loadmat(data_path)

    P_star = mat['p_star']  # (N, T)
    t_star = mat['t'].flatten()
    X_star = mat['X_star']  # (N, 2)

    x_unique = np.unique(X_star[:, 0])
    y_unique = np.unique(X_star[:, 1])

    nx, ny = len(x_unique), len(y_unique)
    T = len(t_star)

    print(f'  网格尺寸: nx={nx}, ny={ny}, nt={T}')

    p_exact = np.zeros((nx, ny, T))

    for i in range(X_star.shape[0]):
        x_val, y_val = X_star[i]
        ix = np.argmin(np.abs(x_unique - x_val))
        iy = np.argmin(np.abs(y_unique - y_val))
        p_exact[ix, iy, :] = P_star[i, :]

    XX, YY, TT = np.meshgrid(x_unique, y_unique, t_star, indexing='ij')

    print(f'  x: [{x_unique.min():.3f}, {x_unique.max():.3f}]')
    print(f'  y: [{y_unique.min():.3f}, {y_unique.max():.3f}]')
    print(f'  t: [{t_star.min():.3f}, {t_star.max():.3f}]')

    return p_exact, XX, YY, TT, x_unique, y_unique


def main():
    parser = argparse.ArgumentParser('NS Equation: Multi-model Error Comparison')
    parser.add_argument('--models', nargs='+', default=['PINN', 'QRes', 'FLS'],
                        help='模型列表')
    parser.add_argument('--model_paths', nargs='*', default=None,
                        help='模型权重路径')
    parser.add_argument('--data_path', type=str, default='./cylinder_nektar_wake.mat')
    parser.add_argument('--time_idx', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./plots')
    parser.add_argument('--filename', type=str, default='ns_pressure_errors.pdf')
    parser.add_argument('--batch_size', type=int, default=1000)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    print(f'加载数据: {args.data_path}')
    p_exact, XX, YY, TT, x_star, y_star = load_ns_data(args.data_path)

    nx, ny, nt = p_exact.shape
    time_idx = min(args.time_idx, nt - 1)

    x_test = XX[:, :, time_idx].flatten()
    y_test = YY[:, :, time_idx].flatten()
    t_test = TT[:, :, time_idx].flatten()

    print(f'  时间切片: t={t_test[0]:.3f} (索引={time_idx})')
    print(f'  测试点数: {len(x_test)}')

    models = args.models
    model_paths = args.model_paths or [f'./results/ns_{m}_point.pt' for m in models]

    if len(model_paths) != len(models):
        raise ValueError('models 与 model_paths 数量不匹配')

    errors = []
    metrics = []
    p_exact_slice = p_exact[:, :, time_idx]

    # 逐模型预测
    for model_name, model_path in zip(models, model_paths):
        if not os.path.exists(model_path):
            print(f'⚠️  跳过 {model_name}: 文件不存在')
            continue

        try:
            print(f'\n加载模型: {model_name}')
            model = build_model(model_name, args.device)
            state = torch.load(model_path, map_location=args.device)
            model.load_state_dict(state)

            if model_name != 'KAN':
                model.eval()

            # 准备输入
            if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only']:
                test_data = np.stack([x_test, y_test, t_test], axis=-1)
                test_data = make_time_sequence(test_data, num_step=5, step=1e-4)
                res_test = torch.tensor(test_data, dtype=torch.float32,
                                        requires_grad=True).to(args.device)
                x_test_t = res_test[..., 0:1]
                y_test_t = res_test[..., 1:2]
                t_test_t = res_test[..., 2:3]
            else:
                x_test_t = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32,
                                        requires_grad=True).to(args.device)
                y_test_t = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32,
                                        requires_grad=True).to(args.device)
                t_test_t = torch.tensor(t_test.reshape(-1, 1), dtype=torch.float32,
                                        requires_grad=True).to(args.device)

            # 预测
            p_pred = predict_in_batches(model, model_name, x_test_t, y_test_t, t_test_t,
                                        args.batch_size)

            p_pred_grid = p_pred.reshape(nx, ny)
            err_grid = np.abs(p_exact_slice - p_pred_grid)
            errors.append(err_grid)

            # 计算误差
            exact_flat = p_exact_slice.flatten()
            pred_flat = p_pred_grid.flatten()
            rl2 = np.linalg.norm(exact_flat - pred_flat) / np.linalg.norm(exact_flat)
            rl1 = np.mean(np.abs(exact_flat - pred_flat)) / np.mean(np.abs(exact_flat))
            metrics.append((model_name, rl1, rl2))

            print(f'  ✓ L1={rl1:.4e}, L2={rl2:.4e}')

            # 清理内存
            del model, state, x_test_t, y_test_t, t_test_t, p_pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f'  ✗ {model_name} 失败: {e}')
            continue

    if not errors:
        print('❌ 没有成功的模型')
        return

    # ============ 绘图（每个子图独立颜色条）============
    n_models = len(errors)
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / 3))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    extent = [x_star.min(), x_star.max(), y_star.min(), y_star.max()]

    for idx, (err_grid, (model_name, rl1, rl2)) in enumerate(zip(errors, metrics)):
        ax = axes[idx]

        # 绘制误差热图
        im = ax.imshow(err_grid.T, extent=extent, origin='lower',
                       aspect='auto', cmap='jet', interpolation='bilinear')

        # ✅ 每个子图独立颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.set_title(f'{model_name}', fontsize=14, pad=10)
        ax.tick_params(labelsize=11)

    # 隐藏多余的子图
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    save_path = os.path.join(args.output_dir, args.filename)
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    plt.close()

    print(f'\n✅ 完成! 保存至: {save_path}')
    print('\n📊 误差统计 (按 L2 排序):')
    for name, rl1, rl2 in sorted(metrics, key=lambda x: x[2]):
        print(f'  {name:20s} L1={rl1:.4e}  L2={rl2:.4e}')


if __name__ == '__main__':
    main()
