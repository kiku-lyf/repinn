import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS
from tqdm import tqdm
import argparse
import numpy as np
import scipy.io
import copy

from models1 import QRes, KAN, FLS, PINN, PINNsFormer,PINNsFormer_Enc_Only,SetPINN


def get_model(args):
    model_dict = {
        'PINN': PINN,
        'QRes': QRes,
        'FLS': FLS,
        'KAN': KAN,
        'PINNsFormer': PINNsFormer,
        'PINNsFormer_Enc_Only':PINNsFormer_Enc_Only,
        'SetPINN':SetPINN,
    }
    return model_dict[args.model]


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, dim)
    for i in range(num_step):
        src[:, i, -1] += step * i
    return src


seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser('Training Point Optimization (NS)')
parser.add_argument('--model', type=str, default='PINN')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
parser.add_argument('--n_train', type=int, default=2500, help='训练点数量')
args = parser.parse_args()
device = args.device

# ==================== 加载 NS 数据 ====================
data = scipy.io.loadmat('./cylinder_nektar_wake.mat')
U_star = data['U_star']  # N x 2 x T
P_star = data['p_star']  # N x T
t_star = data['t']  # T x 1
X_star = data['X_star']  # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data
XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
TT = np.tile(t_star, (1, N)).T  # N x T

UU = U_star[:, 0, :]  # N x T
VV = U_star[:, 1, :]  # N x T
PP = P_star  # N x T

x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1

u = UU.flatten()[:, None]  # NT x 1
v = VV.flatten()[:, None]  # NT x 1
p = PP.flatten()[:, None]  # NT x 1

# 采样训练数据
idx = np.random.choice(N * T, args.n_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
u_train = u[idx, :]
v_train = v[idx, :]

# ==================== 根据模型类型准备数据 ====================
if args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
    # ✅ 正确的方式：先合并为 (N, 3)，然后应用 make_time_sequence
    coords = np.hstack([x_train, y_train, t_train])  # (N, 3)
    coords_seq = make_time_sequence(coords, num_step=5, step=1e-4)  # (N, L, 3)
    # 分别提取 x, y, t
    x_train = coords_seq[:, :, 0:1]  # (N, L, 1)
    y_train = coords_seq[:, :, 1:2]  # (N, L, 1)
    t_train = coords_seq[:, :, 2:3]  # (N, L, 1)
    x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True).to(device)
    t_train = torch.tensor(t_train, dtype=torch.float32, requires_grad=True).to(device)
else:
    x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True).to(device)
    t_train = torch.tensor(t_train, dtype=torch.float32, requires_grad=True).to(device)

# u_train 和 v_train 不需要 requires_grad（它们是标签数据）
u_train = torch.tensor(u_train, dtype=torch.float32).to(device)
v_train = torch.tensor(v_train, dtype=torch.float32).to(device)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# ==================== 构建模型 ====================
if args.model == 'KAN':
    model = get_model(args).Model(width=[3, 5, 5, 2], grid=5, k=3, grid_eps=1.0,
                                  noise_scale_base=0.25, device=device).to(device)
elif args.model == 'QRes':
    model = get_model(args).Model(in_dim=3, hidden_dim=256, out_dim=2, num_layer=4).to(device)
    model.apply(init_weights)
elif args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
    # ✅ 参数顺序: (d_out, d_model, d_hidden, N, heads)
    model = get_model(args).Model(d_out=2, d_model=32, d_hidden=512, N=1, heads=2).to(device)
    model.apply(init_weights)
elif args.model == 'SetPINN':
    model = get_model(args).Model(in_dim=3, hidden_dim=32, out_dim=2, num_layer=3).to(device)
else:
    model = get_model(args).Model(in_dim=3, hidden_dim=512, out_dim=2, num_layer=4).to(device)
    model.apply(init_weights)

optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

print(model)
print(f"Total parameters: {get_n_params(model)}")

loss_track = []
nu = 0.01  # 运动粘度系数

# ==================== 训练循环 ====================
for epoch in tqdm(range(args.epochs)):
    def closure():
        if args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
            psi_and_p = model(x_train, y_train, t_train)
            psi = psi_and_p[:, :, 0:1]  # (N, L, 1)
            p = psi_and_p[:, :, 1:2]  # (N, L, 1)

            # 直接对3D张量计算梯度
            u = torch.autograd.grad(psi, y_train, grad_outputs=torch.ones_like(psi),
                                    retain_graph=True, create_graph=True)[0]
            v = -torch.autograd.grad(psi, x_train, grad_outputs=torch.ones_like(psi),
                                     retain_graph=True, create_graph=True)[0]

            u_t = torch.autograd.grad(u, t_train, grad_outputs=torch.ones_like(u),
                                      retain_graph=True, create_graph=True)[0]
            u_x = torch.autograd.grad(u, x_train, grad_outputs=torch.ones_like(u),
                                      retain_graph=True, create_graph=True)[0]
            u_y = torch.autograd.grad(u, y_train, grad_outputs=torch.ones_like(u),
                                      retain_graph=True, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_train, grad_outputs=torch.ones_like(u_x),
                                       retain_graph=True, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y, y_train, grad_outputs=torch.ones_like(u_y),
                                       retain_graph=True, create_graph=True)[0]

            v_t = torch.autograd.grad(v, t_train, grad_outputs=torch.ones_like(v),
                                      retain_graph=True, create_graph=True)[0]
            v_x = torch.autograd.grad(v, x_train, grad_outputs=torch.ones_like(v),
                                      retain_graph=True, create_graph=True)[0]
            v_y = torch.autograd.grad(v, y_train, grad_outputs=torch.ones_like(v),
                                      retain_graph=True, create_graph=True)[0]
            v_xx = torch.autograd.grad(v_x, x_train, grad_outputs=torch.ones_like(v_x),
                                       retain_graph=True, create_graph=True)[0]
            v_yy = torch.autograd.grad(v_y, y_train, grad_outputs=torch.ones_like(v_y),
                                       retain_graph=True, create_graph=True)[0]

            p_x = torch.autograd.grad(p, x_train, grad_outputs=torch.ones_like(p),
                                      retain_graph=True, create_graph=True)[0]
            p_y = torch.autograd.grad(p, y_train, grad_outputs=torch.ones_like(p),
                                      retain_graph=True, create_graph=True)[0]

            f_u = u_t + (u * u_x + v * u_y) + p_x - nu * (u_xx + u_yy)
            f_v = v_t + (u * v_x + v * v_y) + p_y - nu * (v_xx + v_yy)

            # ✅ 只有 u 和 v 的数据损失，没有压力损失
            loss_data = torch.mean((u[:, 0] - u_train.squeeze()) ** 2) + \
                        torch.mean((v[:, 0] - v_train.squeeze()) ** 2)
            loss_pde = torch.mean(f_u ** 2) + torch.mean(f_v ** 2)
            loss = loss_data + loss_pde

            loss_track.append([loss_data.item(), loss_pde.item()])

            optim.zero_grad()
            loss.backward()
            return loss

        else:
            psi_and_p = model(x_train, y_train, t_train)
            psi = psi_and_p[:, 0:1]
            p = psi_and_p[:, 1:2]

            u = torch.autograd.grad(psi, y_train, grad_outputs=torch.ones_like(psi),
                                    retain_graph=True, create_graph=True)[0]
            v = -torch.autograd.grad(psi, x_train, grad_outputs=torch.ones_like(psi),
                                     retain_graph=True, create_graph=True)[0]

            u_t = torch.autograd.grad(u, t_train, grad_outputs=torch.ones_like(u),
                                      retain_graph=True, create_graph=True)[0]
            u_x = torch.autograd.grad(u, x_train, grad_outputs=torch.ones_like(u),
                                      retain_graph=True, create_graph=True)[0]
            u_y = torch.autograd.grad(u, y_train, grad_outputs=torch.ones_like(u),
                                      retain_graph=True, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_train, grad_outputs=torch.ones_like(u_x),
                                       retain_graph=True, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y, y_train, grad_outputs=torch.ones_like(u_y),
                                       retain_graph=True, create_graph=True)[0]

            v_t = torch.autograd.grad(v, t_train, grad_outputs=torch.ones_like(v),
                                      retain_graph=True, create_graph=True)[0]
            v_x = torch.autograd.grad(v, x_train, grad_outputs=torch.ones_like(v),
                                      retain_graph=True, create_graph=True)[0]
            v_y = torch.autograd.grad(v, y_train, grad_outputs=torch.ones_like(v),
                                      retain_graph=True, create_graph=True)[0]
            v_xx = torch.autograd.grad(v_x, x_train, grad_outputs=torch.ones_like(v_x),
                                       retain_graph=True, create_graph=True)[0]
            v_yy = torch.autograd.grad(v_y, y_train, grad_outputs=torch.ones_like(v_y),
                                       retain_graph=True, create_graph=True)[0]

            p_x = torch.autograd.grad(p, x_train, grad_outputs=torch.ones_like(p),
                                      retain_graph=True, create_graph=True)[0]
            p_y = torch.autograd.grad(p, y_train, grad_outputs=torch.ones_like(p),
                                      retain_graph=True, create_graph=True)[0]

            # u_t = \
            # torch.autograd.grad(u, t_train, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            # u_x = \
            # torch.autograd.grad(u, x_train, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            # u_y = \
            # torch.autograd.grad(u, y_train, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            # u_xx = \
            # torch.autograd.grad(u, x_train, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
            # u_yy = \
            # torch.autograd.grad(u, y_train, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
            #
            # v_t = \
            # torch.autograd.grad(v, t_train, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
            # v_x = \
            # torch.autograd.grad(v, x_train, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
            # v_y = \
            # torch.autograd.grad(v, y_train, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
            # v_xx = \
            # torch.autograd.grad(v, x_train, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
            # v_yy = \
            # torch.autograd.grad(v, y_train, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
            #
            # p_x = \
            # torch.autograd.grad(p, x_train, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
            # p_y = \
            # torch.autograd.grad(p, y_train, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

            f_u = u_t + (u * u_x + v * u_y) + p_x - nu * (u_xx + u_yy)
            f_v = v_t + (u * v_x + v * v_y) + p_y - nu * (v_xx + v_yy)

            # ✅ 只有 u 和 v 的数据损失，没有压力损失
            loss_data = torch.mean((u - u_train) ** 2) + torch.mean((v - v_train) ** 2)
            loss_pde = torch.mean(f_u ** 2) + torch.mean(f_v ** 2)
            loss = loss_data + loss_pde

            loss_track.append([loss_data.item(), loss_pde.item()])

            optim.zero_grad()
            loss.backward()
            return loss


    optim.step(closure)

    # ==================== 每50个epoch保存一次checkpoint ====================
    # if (epoch + 1) % 50 == 0:
    #     if not os.path.exists('./checkpoints/'):
    #         os.makedirs('./checkpoints/')
    #     checkpoint_path = f'./checkpoints/ns_{args.model}_point_epoch{epoch + 1}.pt'
    #     torch.save({
    #         'epoch': epoch + 1,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optim.state_dict(),
    #         'loss': loss_track[-1] if loss_track else None,
    #     }, checkpoint_path)
    #     print(f"\n✅ Checkpoint saved at epoch {epoch + 1}: {checkpoint_path}")

# ==================== 保存最终模型 ====================
if not os.path.exists('./results/'):
    os.makedirs('./results/')

torch.save(model.state_dict(), f'./results/ns_{args.model}_point5.pt')
print(f"✅ Model saved to ./results/ns_{args.model}_point5.pt")

if loss_track:
    print('Loss Data: {:.4e}, Loss PDE: {:.4e}'.format(loss_track[-1][0], loss_track[-1][1]))
    print('Train Loss: {:.4e}'.format(np.sum(loss_track[-1])))

# ==================== 测试评估 ====================
snap = np.array([100])
x_star_np = X_star[:, 0:1]
y_star_np = X_star[:, 1:2]
t_star_np = TT[:, snap]

u_star = U_star[:, 0, snap].flatten()
v_star = U_star[:, 1, snap].flatten()
p_star = P_star[:, snap].flatten()

# 准备测试数据
if args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
    coords_test = np.hstack([x_star_np, y_star_np, t_star_np])  # (N, 3)
    coords_test_seq = make_time_sequence(coords_test, num_step=5, step=1e-4)  # (N, L, 3)
    x_star = torch.tensor(coords_test_seq[:, :, 0:1], dtype=torch.float32, requires_grad=True).to(device)
    y_star = torch.tensor(coords_test_seq[:, :, 1:2], dtype=torch.float32, requires_grad=True).to(device)
    t_star_test = torch.tensor(coords_test_seq[:, :, 2:3], dtype=torch.float32, requires_grad=True).to(device)
else:
    x_star = torch.tensor(x_star_np, dtype=torch.float32, requires_grad=True).to(device)
    y_star = torch.tensor(y_star_np, dtype=torch.float32, requires_grad=True).to(device)
    t_star_test = torch.tensor(t_star_np, dtype=torch.float32, requires_grad=True).to(device)

# ==================== 前向传播（需要梯度计算速度）====================
with torch.enable_grad():
    psi_and_p = model(x_star, y_star, t_star_test)

    if args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
        psi = psi_and_p[:, 0, 0:1]  # 取第一个时间步
        p_pred_tensor = psi_and_p[:, 0, 1:2]
        y_star_grad = y_star[:, 0, :]
        x_star_grad = x_star[:, 0, :]
    else:
        psi = psi_and_p[:, 0:1]
        p_pred_tensor = psi_and_p[:, 1:2]
        y_star_grad = y_star
        x_star_grad = x_star

    u_pred_tensor = torch.autograd.grad(psi, y_star_grad, grad_outputs=torch.ones_like(psi),
                                        retain_graph=True, create_graph=False)[0]
    v_pred_tensor = -torch.autograd.grad(psi, x_star_grad, grad_outputs=torch.ones_like(psi),
                                         retain_graph=True, create_graph=False)[0]

    u_pred = u_pred_tensor.cpu().detach().numpy().flatten()
    v_pred = v_pred_tensor.cpu().detach().numpy().flatten()
    p_pred = p_pred_tensor.cpu().detach().numpy().flatten()

# 误差计算
error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

print("\n" + "=" * 50)
print("L2 Relative Errors:")
print(f"  u: {error_u:.6f}")
print(f"  v: {error_v:.6f}")
print(f"  p: {error_p:.6f}")
print("=" * 50)

# ==================== 可视化 ====================
if not os.path.exists('./plots'):
    os.makedirs('./plots')

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))

# 重塑为网格形状（假设是 50x100 的网格）
p_star_grid = p_star.reshape(50, 100)
p_pred_grid = p_pred.reshape(50, 100)

im0 = axes[0].imshow(p_star_grid, extent=[-3, 8, -2, 2], aspect='auto', cmap='jet', origin='lower')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Exact p(x,y,t)')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(p_pred_grid, extent=[-3, 8, -2, 2], aspect='auto', cmap='jet', origin='lower')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title(f'Predicted p - {args.model}')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(np.abs(p_pred_grid - p_star_grid), extent=[-3, 8, -2, 2], aspect='auto', cmap='jet',
                     origin='lower')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('Absolute Error')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig(f'./plots/ns_{args.model}_comparison5.pdf', dpi=300, bbox_inches='tight')
print(f"✅ Plots saved to ./plots/ns_{args.model}_comparison5.pdf")
plt.close()


