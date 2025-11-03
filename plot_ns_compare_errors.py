# #!/usr/bin/env python3
# """
# 同时对比多个模型在NS方程上的压力场误差热图
#
# 特性：
# - 支持传入多个模型名与模型权重路径
# - 从.mat文件加载真实解数据
# - 统一色标范围，便于横向比较
# - 标题中显示相对L1/L2误差
# """
#
# import argparse
# import os
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import scipy.io
#
# from models1 import QRes, KAN, FLS, PINN, PINNsFormer
#
#
# def predict_ns_model(model_name, model_path, x_test, y_test, t_test, device='cpu'):
#     """
#     使用训练好的模型进行预测（只预测压力 p）
#     """
#     # 根据模型类型构建（输出维度 = 2: ψ 和 p）
#     if model_name == 'PINN':
#         model = PINN.Model(in_dim=3, hidden_dim=512, out_dim=2, num_layer=4).to(device)
#
#     elif model_name == 'QRes':
#         model = QRes.Model(in_dim=3, hidden_dim=256, out_dim=2, num_layer=4).to(device)
#
#     elif model_name == 'FLS':
#         model = FLS.Model(in_dim=3, hidden_dim=512, out_dim=2, num_layer=4).to(device)
#
#     elif model_name == 'KAN':
#         model = KAN.Model(width=[3, 5, 5, 2], grid=5, k=3, grid_eps=1.0,
#                           noise_scale_base=0.25, device=device).to(device)
#
#     elif model_name == 'PINNsFormer':
#         model = PINNsFormer.Model(in_dim=3, hidden_dim=32, out_dim=2, num_layer=1).to(device)
#
#     else:
#         raise ValueError(f"未知模型: {model_name}")
#
#     # 加载模型权重
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#
#     # 准备输入数据
#     x_tensor = torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1).to(device)
#     y_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)
#     t_tensor = torch.tensor(t_test, dtype=torch.float32).reshape(-1, 1).to(device)
#
#     # 根据模型类型准备输入
#     if model_name == 'PINNsFormer':
#         X_test = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)  # (N, 3)
#         X_test = X_test.unsqueeze(1)  # (N, 1, 3) - PINNsFormer 需要序列维度
#     else:
#         X_test = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)  # (N, 3)
#
#     # 预测
#     with torch.no_grad():
#         output = model(X_test)
#
#         # 处理 PINNsFormer 的输出
#         if model_name == 'PINNsFormer':
#             output = output.squeeze(1)  # (N, 2)
#
#         # 提取压力（第二个输出）
#         p_pred = output[:, 1].cpu().numpy()
#
#     return p_pred
#
#
# def load_ns_data(data_path=None):
#     """加载 NS 方程的真实解数据"""
#     if data_path is None or not os.path.exists(data_path):
#         return None, None, None, None, None, None
#
#     try:
#         mat = scipy.io.loadmat(data_path)
#
#         P_star = mat['p_star']  # (N, T)
#         t_star = mat['t'].flatten()  # (T,)
#         X_star = mat['X_star']  # (N, 2)
#
#         N = X_star.shape[0]
#         T = len(t_star)
#
#         # 提取坐标
#         x_unique = np.unique(X_star[:, 0])
#         y_unique = np.unique(X_star[:, 1])
#
#         nx = len(x_unique)
#         ny = len(y_unique)
#
#         print(f"检测到的网格尺寸: nx={nx}, ny={ny}, nt={T}, 总点数={N}")
#
#         # 准备压力数组
#         p_exact = np.zeros((nx, ny, T))
#
#         # 填充数据
#         for i in range(N):
#             x_val = X_star[i, 0]
#             y_val = X_star[i, 1]
#
#             ix = np.argmin(np.abs(x_unique - x_val))
#             iy = np.argmin(np.abs(y_unique - y_val))
#
#             p_exact[ix, iy, :] = P_star[i, :]
#
#         # 创建网格
#         x_star = x_unique
#         y_star = y_unique
#         XX, YY, TT = np.meshgrid(x_star, y_star, t_star, indexing='ij')
#
#         print(f"✅ 成功加载 NS 数据:")
#         print(f"  原始数据: N={N}, T={T}")
#         print(f"  网格尺寸: nx={nx}, ny={ny}, nt={T}")
#         print(f"  p_exact.shape = {p_exact.shape}")
#         print(f"  x: [{x_star[0]:.3f}, {x_star[-1]:.3f}], {len(x_star)} 点")
#         print(f"  y: [{y_star[0]:.3f}, {y_star[-1]:.3f}], {len(y_star)} 点")
#         print(f"  t: [{t_star[0]:.3f}, {t_star[-1]:.3f}], {len(t_star)} 点")
#
#         return p_exact, XX, YY, TT, x_star, y_star
#
#     except Exception as e:
#         print(f"❌ 错误: 无法加载数据文件 {data_path}: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return None, None, None, None, None, None
#
#
# def main():
#     parser = argparse.ArgumentParser('Compare multiple models: NS pressure field errors')
#     parser.add_argument('--models', nargs='+', required=False,
#                         default=['PINN', 'QRes', 'FLS'],
#                         help='模型名称列表，如: PINN KAN QRes FLS PINNsFormer')
#     parser.add_argument('--model_paths', nargs='*', default=None,
#                         help='模型权重列表（与models一一对应）。若不提供则按默认规则从./results推断')
#     parser.add_argument('--data_path', type=str, default='./cylinder_nektar_wake.mat',
#                         help='NS真实解数据文件路径（.mat格式）')
#     parser.add_argument('--time_idx', type=int, default=-1,
#                         help='时间切片索引（-1表示最后一个时刻）')
#     parser.add_argument('--device', type=str, default='cpu')
#     parser.add_argument('--output_dir', type=str, default='./plots')
#     parser.add_argument('--filename', type=str, default='ns_pressure_errors.pdf')
#     parser.add_argument('--cols', type=int, default=3, help='每行显示的子图列数')
#
#     args = parser.parse_args()
#
#     os.makedirs(args.output_dir, exist_ok=True)
#
#     models = args.models
#     # 推断模型路径
#     if args.model_paths is None or len(args.model_paths) == 0:
#         model_paths = [f'./results/ns_{m}_point.pt' for m in models]
#     else:
#         if len(args.model_paths) != len(models):
#             raise ValueError('models 与 model_paths 数量不一致')
#         model_paths = args.model_paths
#
#     # 加载真实解数据
#     print(f"加载真实解数据: {args.data_path}")
#     p_exact, XX, YY, TT, x_star, y_star = load_ns_data(args.data_path)
#
#     if p_exact is None:
#         print("❌ 无法加载真实解数据，程序退出")
#         return
#
#     nx, ny, nt = p_exact.shape
#     time_idx = args.time_idx if args.time_idx >= 0 else nt - 1
#
#     # 提取测试数据
#     x_test = XX[:, :, time_idx].flatten()
#     y_test = YY[:, :, time_idx].flatten()
#     t_test = TT[:, :, time_idx].flatten()
#
#     print(f"使用时间切片: t={t_test[0]:.3f} (索引={time_idx})")
#
#     # 存储所有模型的误差和指标
#     errors = []
#     metrics = []  # (name, rl1, rl2)
#
#     # 获取真实解在指定时刻的数据
#     p_exact_slice = p_exact[:, :, time_idx]
#
#     # 逐模型预测
#     for model_name, model_path in zip(models, model_paths):
#         if not os.path.exists(model_path):
#             print(f'⚠️  模型权重不存在: {model_path}，跳过该模型')
#             continue
#
#         print(f"\n{'=' * 60}")
#         print(f'加载模型: {model_name} | {model_path}')
#         print('=' * 60)
#
#         try:
#             p_pred = predict_ns_model(
#                 model_name, model_path,
#                 x_test, y_test, t_test,
#                 device=args.device
#             )
#
#             # 重塑为网格
#             p_pred_grid = p_pred.reshape(nx, ny)
#
#             # 计算误差
#             err_grid = np.abs(p_exact_slice - p_pred_grid)
#             errors.append(err_grid)
#
#             # 计算相对误差
#             exact_flat = p_exact_slice.flatten()
#             pred_flat = p_pred_grid.flatten()
#             rl2 = np.linalg.norm(exact_flat - pred_flat, 2) / np.linalg.norm(exact_flat, 2)
#             rl1 = np.mean(np.abs(exact_flat - pred_flat)) / np.mean(np.abs(exact_flat))
#             metrics.append((model_name, rl1, rl2))
#
#             print(f"✅ 相对 L1 误差: {rl1:.6e}")
#             print(f"✅ 相对 L2 误差: {rl2:.6e}")
#
#         except Exception as e:
#             print(f"❌ 错误: 处理模型 {model_name} 时出错: {str(e)}")
#             import traceback
#             traceback.print_exc()
#
#     if len(errors) == 0:
#         print('无可用模型，结束。')
#         return
#
#     # 统一色标范围
#     vmax = max(np.max(err) for err in errors)
#
#     # 画图
#     n = len(errors)
#     cols = max(1, args.cols)
#     rows = (n + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
#
#     extent = [x_star.min(), x_star.max(), y_star.min(), y_star.max()]
#
#     idx = 0
#     for r in range(rows):
#         for c in range(cols):
#             ax = axes[r][c]
#             if idx >= n:
#                 ax.axis('off')
#                 continue
#
#             err = errors[idx]
#             name, rl1, rl2 = metrics[idx]
#
#             # 使用 imshow 绘制热力图
#             im = ax.imshow(err.T, extent=extent, origin='lower',
#                            aspect='auto', cmap='jet', vmin=0.0, vmax=vmax)
#
#             ax.set_xlabel('$x$')
#             ax.set_ylabel('$y$')
#             ax.set_title(f'NS Pressure Error ({name})\nL1: {rl1:.2e}, L2: {rl2:.2e}')
#             idx += 1
#
#     # 统一色条
#     fig.subplots_adjust(right=0.88, wspace=0.25, hspace=0.35)
#     cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
#     fig.colorbar(im, cax=cbar_ax, label='|p_exact - p_pred|')
#
#     save_path = os.path.join(args.output_dir, args.filename)
#     plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
#     plt.close(fig)
#
#     print(f"\n{'=' * 60}")
#     print('✅ 完成，多模型误差对比图已保存:', save_path)
#     print('=' * 60)
#     print('相对误差总结:')
#     print('=' * 60)
#     for name, rl1, rl2 in sorted(metrics, key=lambda x: x[2]):
#         print(f'  {name:15s}: L1={rl1:.6e}, L2={rl2:.6e}')
#     print('=' * 60)
#
#
# if __name__ == '__main__':
#     main()


# !/usr/bin/env python3
"""
同时对比多个模型在NS方程上的压力场误差热图

特性：
- 支持传入多个模型名与模型权重路径
- 从.mat文件加载真实解数据
- 统一色标范围，便于横向比较
- 标题中显示相对L1/L2误差
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io

from models1 import QRes, KAN, FLS, PINN, PINNsFormer,PINNsFormer_Enc_Only


def make_time_sequence(src, num_step=5, step=1e-4):
    """为 PINNsFormer 创建时间序列"""
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, ...)
    for i in range(num_step):
        src[:, i, -1] += step * i
    return src


def predict_ns_model(model_name, model_path, x_test, y_test, t_test, device='cpu'):
    """
    使用训练好的模型进行预测（只预测压力 p）

    参数:
        x_test, y_test, t_test: numpy arrays, shape (N,) 或 (N, 1)
    """
    # 根据模型类型构建（输出维度 = 2: ψ 和 p）
    if model_name == 'PINN':
        model = PINN.Model(in_dim=3, hidden_dim=512, out_dim=2, num_layer=4).to(device)

    elif model_name == 'QRes':
        model = QRes.Model(in_dim=3, hidden_dim=256, out_dim=2, num_layer=4).to(device)

    elif model_name == 'FLS':
        model = FLS.Model(in_dim=3, hidden_dim=512, out_dim=2, num_layer=4).to(device)

    elif model_name == 'KAN':
        model = KAN.Model(width=[3, 5, 5, 2], grid=5, k=3, grid_eps=1.0,
                          noise_scale_base=0.25, device=device).to(device)

    elif model_name == 'PINNsFormer':
          # 到底是64还是512
        model = PINNsFormer.Model(d_out=2, d_hidden=64, d_model=32, N=1, heads=2).to(device)

    elif model_name == 'PINNsFormer_Enc_Only':
        model = PINNsFormer_Enc_Only.Model(d_out=2, d_hidden=512, d_model=32, N=1, heads=2).to(device)

    else:
        raise ValueError(f"未知模型: {model_name}")

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 准备输入数据
    if model_name == 'PINNsFormer' or model_name == 'PINNsFormer_Enc_Only':
        # PINNsFormer 需要特殊的时间序列格式
        x_np = x_test.reshape(-1, 1)
        y_np = y_test.reshape(-1, 1)
        t_np = t_test.reshape(-1, 1)

        # 扩展为时间序列
        x_seq = np.expand_dims(np.tile(x_np, (1, 5)), -1)  # (N, 5, 1)
        y_seq = np.expand_dims(np.tile(y_np, (1, 5)), -1)  # (N, 5, 1)
        t_seq = make_time_sequence(t_np, num_step=5, step=1e-2)  # (N, 5, 1)

        x_tensor = torch.tensor(x_seq, dtype=torch.float32, requires_grad=True).to(device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32, requires_grad=True).to(device)
        t_tensor = torch.tensor(t_seq, dtype=torch.float32, requires_grad=True).to(device)

        # 预测
        with torch.no_grad():
            psi_and_p = model(x_tensor, y_tensor, t_tensor)  # (N, 5, 2)

            # 计算速度（需要梯度）
            psi = psi_and_p[:, :, 0:1]
            p_pred = psi_and_p[:, :, 1:2]

            # 提取第一个时间步的压力
            p_pred = p_pred[:, 0, 0].cpu().numpy()

    else:
        # 其他模型使用标准格式
        x_tensor = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        y_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
        t_tensor = torch.tensor(t_test.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)

        # 预测
        with torch.no_grad():
            psi_and_p = model(x_tensor, y_tensor, t_tensor)  # (N, 2)

            # 提取压力（第二个输出）
            p_pred = psi_and_p[:, 1].cpu().numpy()

    return p_pred


def load_ns_data(data_path=None):
    """加载 NS 方程的真实解数据"""
    if data_path is None or not os.path.exists(data_path):
        return None, None, None, None, None, None

    try:
        mat = scipy.io.loadmat(data_path)

        P_star = mat['p_star']  # (N, T)
        t_star = mat['t'].flatten()  # (T,)
        X_star = mat['X_star']  # (N, 2)

        N = X_star.shape[0]
        T = len(t_star)

        # 提取坐标
        x_unique = np.unique(X_star[:, 0])
        y_unique = np.unique(X_star[:, 1])

        nx = len(x_unique)
        ny = len(y_unique)

        print(f"检测到的网格尺寸: nx={nx}, ny={ny}, nt={T}, 总点数={N}")

        # 准备压力数组
        p_exact = np.zeros((nx, ny, T))

        # 填充数据
        for i in range(N):
            x_val = X_star[i, 0]
            y_val = X_star[i, 1]

            ix = np.argmin(np.abs(x_unique - x_val))
            iy = np.argmin(np.abs(y_unique - y_val))

            p_exact[ix, iy, :] = P_star[i, :]

        # 创建网格
        x_star = x_unique
        y_star = y_unique
        XX, YY, TT = np.meshgrid(x_star, y_star, t_star, indexing='ij')

        print(f"✅ 成功加载 NS 数据:")
        print(f"  原始数据: N={N}, T={T}")
        print(f"  网格尺寸: nx={nx}, ny={ny}, nt={T}")
        print(f"  p_exact.shape = {p_exact.shape}")
        print(f"  x: [{x_star[0]:.3f}, {x_star[-1]:.3f}], {len(x_star)} 点")
        print(f"  y: [{y_star[0]:.3f}, {y_star[-1]:.3f}], {len(y_star)} 点")
        print(f"  t: [{t_star[0]:.3f}, {t_star[-1]:.3f}], {len(t_star)} 点")

        return p_exact, XX, YY, TT, x_star, y_star

    except Exception as e:
        print(f"❌ 错误: 无法加载数据文件 {data_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None


def main():
    parser = argparse.ArgumentParser('Compare multiple models: NS pressure field errors')
    parser.add_argument('--models', nargs='+', required=False,
                        default=['PINN', 'QRes', 'FLS'],
                        help='模型名称列表，如: PINN KAN QRes FLS PINNsFormer')
    parser.add_argument('--model_paths', nargs='*', default=None,
                        help='模型权重列表（与models一一对应）。若不提供则按默认规则从./results推断')
    parser.add_argument('--data_path', type=str, default='./cylinder_nektar_wake.mat',
                        help='NS真实解数据文件路径（.mat格式）')
    parser.add_argument('--time_idx', type=int, default=100,
                        help='时间切片索引（默认100，对应训练代码中的snap）')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='./plots')
    parser.add_argument('--filename', type=str, default='ns_pressure_errors.pdf')
    parser.add_argument('--cols', type=int, default=3, help='每行显示的子图列数')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models = args.models
    # 推断模型路径
    if args.model_paths is None or len(args.model_paths) == 0:
        model_paths = [f'./results/ns_{m}_point.pt' for m in models]
    else:
        if len(args.model_paths) != len(models):
            raise ValueError('models 与 model_paths 数量不一致')
        model_paths = args.model_paths

    # 加载真实解数据
    print(f"加载真实解数据: {args.data_path}")
    p_exact, XX, YY, TT, x_star, y_star = load_ns_data(args.data_path)

    if p_exact is None:
        print("❌ 无法加载真实解数据，程序退出")
        return

    nx, ny, nt = p_exact.shape
    time_idx = args.time_idx

    if time_idx >= nt:
        print(f"⚠️  警告: time_idx={time_idx} 超出范围，使用最后一个时刻 {nt - 1}")
        time_idx = nt - 1

    # 提取测试数据（匹配训练代码中的格式）
    x_test = XX[:, :, time_idx].flatten()
    y_test = YY[:, :, time_idx].flatten()
    t_test = TT[:, :, time_idx].flatten()

    print(f"使用时间切片: t={t_test[0]:.3f} (索引={time_idx})")
    print(f"测试点数量: {len(x_test)}")

    # 存储所有模型的误差和指标
    errors = []
    metrics = []  # (name, rl1, rl2)

    # 获取真实解在指定时刻的数据
    p_exact_slice = p_exact[:, :, time_idx]

    # 逐模型预测
    for model_name, model_path in zip(models, model_paths):
        if not os.path.exists(model_path):
            print(f'⚠️  模型权重不存在: {model_path}，跳过该模型')
            continue

        print(f"\n{'=' * 60}")
        print(f'加载模型: {model_name} | {model_path}')
        print('=' * 60)

        try:
            p_pred = predict_ns_model(
                model_name, model_path,
                x_test, y_test, t_test,
                device=args.device
            )

            # 重塑为网格
            p_pred_grid = p_pred.reshape(nx, ny)

            # 计算误差
            err_grid = np.abs(p_exact_slice - p_pred_grid)
            errors.append(err_grid)

            # 计算相对误差
            exact_flat = p_exact_slice.flatten()
            pred_flat = p_pred_grid.flatten()
            rl2 = np.linalg.norm(exact_flat - pred_flat, 2) / np.linalg.norm(exact_flat, 2)
            rl1 = np.mean(np.abs(exact_flat - pred_flat)) / np.mean(np.abs(exact_flat))
            metrics.append((model_name, rl1, rl2))

            print(f"✅ 相对 L1 误差: {rl1:.6e}")
            print(f"✅ 相对 L2 误差: {rl2:.6e}")

        except Exception as e:
            print(f"❌ 错误: 处理模型 {model_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()

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

    extent = [x_star.min(), x_star.max(), y_star.min(), y_star.max()]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx >= n:
                ax.axis('off')
                continue

            err = errors[idx]
            name, rl1, rl2 = metrics[idx]

            # 使用 imshow 绘制热力图
            im = ax.imshow(err.T, extent=extent, origin='lower',
                           aspect='auto', cmap='jet', vmin=0.0, vmax=vmax)

            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_title(f'NS Pressure Error ({name})\nL1: {rl1:.2e}, L2: {rl2:.2e}')
            idx += 1

    # 统一色条
    fig.subplots_adjust(right=0.88, wspace=0.25, hspace=0.35)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='|p_exact - p_pred|')

    save_path = os.path.join(args.output_dir, args.filename)
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"\n{'=' * 60}")
    print('✅ 完成，多模型误差对比图已保存:', save_path)
    print('=' * 60)
    print('相对误差总结:')
    print('=' * 60)
    for name, rl1, rl2 in sorted(metrics, key=lambda x: x[2]):
        print(f'  {name:15s}: L1={rl1:.6e}, L2={rl2:.6e}')
    print('=' * 60)


if __name__ == '__main__':
    main()
