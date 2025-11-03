#!/usr/bin/env python3
"""
同时对比多个模型在KdV方程上的误差热图

特性：
- 支持传入多个模型名与模型权重路径
- 支持从.mat文件加载真实解数据
- 统一色标范围，便于横向比较
- 标题中显示相对L1/L2误差
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io

from plot_utils import load_model_and_predict
from model_dict import get_model


def load_kdv_data(data_path=None):
    """
    加载KdV方程的真实解数据
    
    Args:
        data_path: 数据文件路径（.mat格式）
    
    Returns:
        exact: 真实解数据
        XX, TT: 网格坐标
        x_star, t_star: 坐标轴
    """
    if data_path is None or not os.path.exists(data_path):
        return None, None, None, None, None
    
    try:
        mat = scipy.io.loadmat(data_path)
        
        # 提取解数据（根据用户示例，变量名是 'usol'）
        if 'usol' in mat:
            exact = mat['usol']
        elif 'u' in mat:
            exact = mat['u']
        elif 'uu' in mat:
            exact = mat['uu']
        elif 'solution' in mat:
            exact = mat['solution']
        else:
            raise ValueError(f"无法在 {data_path} 中找到解数据（尝试查找 'usol', 'u', 'uu', 'solution'）")
        
        # 提取网格信息（根据用户示例，变量名是 'x' 和 't'）
        if 'x' in mat and 't' in mat:
            x_star = mat['x'].flatten()
            t_star = mat['t'].flatten()
        elif 'x' in mat and 'tt' in mat:
            x_star = mat['x'].flatten()
            t_star = mat['tt'].flatten()
        else:
            # 根据数据形状推断网格
            if len(exact.shape) == 2:
                n_x, n_t = exact.shape
                # 根据 KdV 方程常见的范围推断
                x_star = np.linspace(-1, 1, n_x)
                t_star = np.linspace(0, 1, n_t)
                print(f"警告: 未找到网格信息，根据数据形状推断: x in [-1,1], t in [0,1]")
            else:
                raise ValueError(f"无法推断网格尺寸，数据形状: {exact.shape}")

        TT, XX = np.meshgrid(t_star, x_star)

        print(f"成功加载数据: exact.shape={exact.shape}, x_star.shape={x_star.shape}, t_star.shape={t_star.shape}")

        # —— 新增：对齐 exact 的轴顺序，让它变成 (len(x), len(t)) ——
        # 期望：行对应 x，列对应 t
        if exact.shape == (t_star.size, x_star.size):
            exact = exact.T  # 现在 exact 变为 (len(x), len(t))
        elif exact.shape != (x_star.size, t_star.size):
            raise ValueError(
                f"解矩阵的形状与网格不匹配: exact={exact.shape}, 期望为 "
                f"({x_star.size}, {t_star.size}) 或 ({t_star.size}, {x_star.size})"
            )

        return exact, XX, TT, x_star, t_star
    except Exception as e:
        print(f"错误: 无法加载数据文件 {data_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def main():
    parser = argparse.ArgumentParser('Compare multiple models: KdV errors')
    parser.add_argument('--models', nargs='+', required=False,
                        default=['PINN'],
                        help='模型名称列表，如: PINN KAN QRes')
    parser.add_argument('--model_paths', nargs='*', default=None,
                        help='模型权重列表（与models一一对应）。若不提供则按默认规则从./results推断')
    parser.add_argument('--data_path', type=str, default='./kdv.mat',
                        help='KdV真实解数据文件路径（.mat格式，默认: ./kdv.mat）')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--grid_size', type=int, default=101)
    parser.add_argument('--x_range', type=float, nargs=2, default=[-1, 1],
                        help='x轴范围，如: -1 1')
    parser.add_argument('--t_range', type=float, nargs=2, default=[0, 1],
                        help='t轴范围，如: 0 1')
    parser.add_argument('--output_dir', type=str, default='./plots')
    parser.add_argument('--filename', type=str, default='kdv_models_errors.pdf')
    parser.add_argument('--cols', type=int, default=3, help='每行显示的子图列数')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    x_range = args.x_range
    t_range = args.t_range

    models = args.models
    # 推断模型路径
    if args.model_paths is None or len(args.model_paths) == 0:
        model_paths = [f'./results/kdv_{m}_point.pt' for m in models]
    else:
        if len(args.model_paths) != len(models):
            raise ValueError('models 与 model_paths 数量不一致')
        model_paths = args.model_paths

    errors = []
    metrics = []  # (name, rl1, rl2)

    XX_ref = None
    TT_ref = None
    exact_ref = None
    x_star_ref = None
    t_star_ref = None

    # 尝试加载真实解数据
    exact_data, XX_data, TT_data, x_star_data, t_star_data = load_kdv_data(args.data_path)
    
    if exact_data is not None:
        print(f'使用真实解数据: {args.data_path}')
        XX_ref, TT_ref = XX_data, TT_data
        exact_ref = exact_data
        x_star_ref, t_star_ref = x_star_data, t_star_data
        # 根据参考数据的网格范围更新 x_range 和 t_range
        x_range = [float(x_star_ref[0]), float(x_star_ref[-1])]
        t_range = [float(t_star_ref[0]), float(t_star_ref[-1])]
        print(f'根据数据文件更新范围: x_range={x_range}, t_range={t_range}')
        use_external_data = True
    else:
        if args.data_path is None:
            print('错误: KdV方程需要参考解数据，请使用--data_path参数指定.mat文件')
            return
        else:
            print(f'错误: 无法加载数据文件 {args.data_path}')
            return

    # 逐模型预测
    for model_name, model_path in zip(models, model_paths):
        if not os.path.exists(model_path):
            print(f'警告: 模型权重不存在: {model_path}，跳过该模型')
            continue

        print(f'加载模型: {model_name} | {model_path}')
        
        if use_external_data:
            # 使用外部数据的网格进行预测（根据用户示例的方式）
            # 按照用户示例：data = np.hstack([XX.reshape(-1,1), TT.reshape(-1,1)])
            data = np.hstack([XX_ref.reshape(-1, 1), TT_ref.reshape(-1, 1)])
            
            # PINNsFormer 需要 make_time_sequence 处理
            if model_name == 'PINNsFormer' or model_name == 'PINNsFormer_Enc_Only':
                from util import make_time_sequence
                # 对于 PINNsFormer，需要对原始数据应用 make_time_sequence
                # 但需要先reshape为 (N, 2) 格式
                res_test = make_time_sequence(data, num_step=5, step=1e-4)
                res_test_tensor = torch.tensor(res_test, dtype=torch.float32, requires_grad=False).to(args.device)
                x_test, t_test = res_test_tensor[:, ..., 0:1], res_test_tensor[:, ..., 1:2]
            else:
                # 其他模型直接使用数据
                res_test_tensor = torch.tensor(data, dtype=torch.float32, requires_grad=False).to(args.device)
                x_test, t_test = res_test_tensor[:, 0:1], res_test_tensor[:, 1:2]
            
            model_module = get_model(type('args', (), {'model': model_name})())
            if model_name == 'KAN':
                model = model_module.Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25, device=args.device).to(args.device)
            elif model_name == 'SetPINN':
                model = model_module.Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=3).to(args.device)
            elif model_name == 'QRes':
                model = model_module.Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(args.device)
            elif model_name == 'PINNsFormer' or model_name == 'PINNsFormer_Enc_Only':
                model = model_module.Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(args.device)
            else:
                model = model_module.Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(args.device)
            
            model.load_state_dict(torch.load(model_path, map_location=args.device))
            if model_name != 'KAN':
                model.eval()
            
            with torch.no_grad():
                pred = model(x_test, t_test)[:, 0:1]
                pred = pred.cpu().numpy().reshape(-1)
            
            XX, TT = XX_ref, TT_ref
        else:
            # 使用自定义网格进行预测
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
                # 如果没有外部数据，需要计算参考解
                print('警告: 未提供参考解数据，无法计算误差')
                continue

        # 根据用户示例，数据是通过 XX.reshape(-1,1) 和 TT.reshape(-1,1) 拼接的
        # 这意味着 pred 的顺序是按照 XX, TT 的 flatten 顺序，即先遍历 x（固定 t），再遍历 t
        # XX 和 TT 是通过 meshgrid(t_star, x_star) 生成的，形状是 (len(x_star), len(t_star))
        # 所以 pred 应该 reshape 为 exact_ref 的形状
        
        # 直接将 pred reshape 为 exact_ref 的形状（因为数据生成顺序一致）
        pred_grid = pred.reshape(exact_ref.shape)
        
        # 误差计算
        err_grid = np.abs(exact_ref - pred_grid)
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
            im = ax.pcolormesh(t_star_ref, x_star_ref, err, cmap='jet', shading='auto', vmin=0.0, vmax=vmax)

            ax.set_xlabel('$t$')
            ax.set_ylabel('$x$')
            ax.set_title(f'KdV ({name}) | L1: {rl1:.2e}, L2: {rl2:.2e}')
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

