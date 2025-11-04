#!/usr/bin/env python3
"""
同时对比多个模型在KdV方程上的误差热图
每个子图独立颜色条，与 Burgers/NS/Allen-Cahn 代码保持完全一致
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
    """构建模型（与其他代码保持一致）"""
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

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return np.concatenate(predictions, axis=0)


def load_kdv_data(data_path):
    """加载 KdV 方程的真实解数据"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'数据文件不存在: {data_path}')

    mat = scipy.io.loadmat(data_path)

    # 提取解数据
    exact = None
    for var_name in ['usol', 'u', 'uu', 'solution']:
        if var_name in mat:
            exact = mat[var_name]
            print(f'  找到解数据变量: {var_name}')
            break

    if exact is None:
        raise ValueError(f"无法在 {data_path} 中找到解数据")

    # 提取网格信息
    x_star = None
    t_star = None

    if 'x' in mat:
        x_star = mat['x'].flatten()
    if 't' in mat:
        t_star = mat['t'].flatten()
    elif 'tt' in mat:
        t_star = mat['tt'].flatten()

    # 如果找不到网格信息，根据数据形状推断
    if x_star is None or t_star is None:
        if len(exact.shape) != 2:
            raise ValueError(f'解数据应为2D数组，实际形状: {exact.shape}')
        n_x, n_t = exact.shape
        x_star = np.linspace(-1, 1, n_x)
        t_star = np.linspace(0, 1, n_t)
        print(f'  警告: 未找到网格信息，根据数据形状推断')

    # 生成网格
    TT, XX = np.meshgrid(t_star, x_star)

    # 对齐数据形状
    if exact.shape == (t_star.size, x_star.size):
        exact = exact.T
        print(f'  数据转置: {(t_star.size, x_star.size)} -> {exact.shape}')
    elif exact.shape != (x_star.size, t_star.size):
        raise ValueError(f"解矩阵的形状与网格不匹配: exact={exact.shape}")

    print(f'  数据加载成功: exact.shape={exact.shape}')
    print(f'  x: [{x_star.min():.3f}, {x_star.max():.3f}], n={len(x_star)}')
    print(f'  t: [{t_star.min():.3f}, {t_star.max():.3f}], n={len(t_star)}')

    return exact, XX, TT, x_star, t_star


def main():
    parser = argparse.ArgumentParser('Compare multiple models: KdV errors')
    parser.add_argument('--models', nargs='+', required=False,
                        default=['PINN', 'KAN', 'QRes'],
                        help='模型名称列表')
    parser.add_argument('--model_paths', nargs='*', default=None,
                        help='模型权重列表')
    parser.add_argument('--data_path', type=str, default='./kdv.mat')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./plots')
    parser.add_argument('--filename', type=str, default='kdv_models_errors.pdf')
    parser.add_argument('--batch_size', type=int, default=1000)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 读取真实解数据
    print(f'加载真实解数据: {args.data_path}')
    try:
        Exact, XX_ref, TT_ref, x_star, t_star = load_kdv_data(args.data_path)
    except Exception as e:
        print(f'错误: 无法加载数据文件: {str(e)}')
        import traceback
        traceback.print_exc()
        return

    # 生成测试数据
    data = np.hstack([XX_ref.reshape(-1, 1), TT_ref.reshape(-1, 1)])
    print(f'  测试数据总点数: {data.shape[0]}')

    models = args.models
    if args.model_paths is None or len(args.model_paths) == 0:
        model_paths = [f'./results/kdv_{m}_point.pt' for m in models]
    else:
        if len(args.model_paths) != len(models):
            raise ValueError('models 与 model_paths 数量不一致')
        model_paths = args.model_paths

    errors = []
    metrics = []

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

            # 准备测试数据
            test_data = data.copy()
            if model_name == 'PINNsFormer' or model_name == 'PINNsFormer_Enc_Only':
                test_data = make_time_sequence(test_data, num_step=5, step=1e-4)

            res_test = torch.tensor(test_data, dtype=torch.float32,
                                    requires_grad=True).to(args.device)
            x_test, t_test = res_test[:, ..., 0:1], res_test[:, ..., 1:2]

            # 预测
            if model_name in ['PINNsFormer', 'PINNsFormer_Enc_Only']:
                pred = predict_in_batches(model, x_test, t_test, args.batch_size)
            else:
                with torch.no_grad():
                    pred = model(x_test, t_test)[:, 0:1]
                    pred = pred.cpu().numpy()

            pred = pred.reshape(-1)
            pred_grid = pred.reshape(Exact.shape)

            # 误差计算
            Exact_flat = Exact.flatten()
            err_grid = np.abs(Exact - pred_grid)
            errors.append(err_grid)

            # 计算指标
            rl2 = np.linalg.norm(Exact_flat - pred, 2) / np.linalg.norm(Exact_flat, 2)
            rl1 = np.mean(np.abs(Exact_flat - pred)) / np.mean(np.abs(Exact_flat))
            metrics.append((model_name, rl1, rl2))

            print(f'  ✓ L1={rl1:.4e}, L2={rl2:.4e}')

            # 释放显存
            del model, state, res_test, x_test, t_test, pred, pred_grid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f'  ✗ {model_name} 失败: {str(e)}')
            import traceback
            traceback.print_exc()
            continue

    if len(errors) == 0:
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

    print(f'\n布局: {n_models}个模型，{n_rows} 行 × {n_cols} 列')

    for idx, (err_grid, (model_name, rl1, rl2)) in enumerate(zip(errors, metrics)):
        ax = axes[idx]

        # 使用 pcolormesh 绘制热图
        mesh = ax.pcolormesh(TT_ref, XX_ref, err_grid, cmap='jet',
                             shading='auto', vmin=0.0)

        # ✅ 每个子图独立颜色条
        cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

        ax.set_xlabel('$t$', fontsize=14)
        ax.set_ylabel('$x$', fontsize=14)
        ax.set_title(f'{model_name}', fontsize=14, pad=10)
        ax.tick_params(labelsize=11)

    # 隐藏多余的子图
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    save_path = os.path.join(args.output_dir, args.filename)
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f'\n✅ 完成! 保存至: {save_path}')
    print('\n📊 误差统计 (按 L2 排序):')
    for name, rl1, rl2 in sorted(metrics, key=lambda x: x[2]):
        print(f'  {name:20s} L1={rl1:.4e}  L2={rl2:.4e}')


if __name__ == '__main__':
    main()
