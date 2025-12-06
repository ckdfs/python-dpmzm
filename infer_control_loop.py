import numpy as np
import torch

from generate_dataset import (
    generate_pilot_tones,
    extract_relative_pilot_features,
)
from test_dpmzm import (
    build_params,
    init_backend,
    photodetect_and_add_noise,
    electrical_spectrum,
    run_dpmzm_model,
)


class BiasControlMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # 必须和 train_mlp.MLPRegressor 完全一致：12 -> 128 -> 128 -> 64 -> 3
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_trained_model(model_path: str = "mlp_bias_control_best.pth", device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_path, map_location=device)
    input_dim = int(checkpoint["input_dim"])
    output_dim = int(checkpoint["output_dim"])

    model = BiasControlMLP(input_dim, output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_mean = torch.from_numpy(checkpoint["X_mean"]).float().to(device)
    X_std = torch.from_numpy(checkpoint["X_std"]).float().to(device)
    Y_mean = torch.from_numpy(checkpoint["Y_mean"]).float().to(device)
    Y_std = torch.from_numpy(checkpoint["Y_std"]).float().to(device)

    return model, device, X_mean, X_std, Y_mean, Y_std


def simulate_measurement(
    V_I_bias: float,
    V_Q_bias: float,
    V_P_bias: float,
    Fs: float = 10e6,
    T_total: float = 5e-3,
    pilot_freqs_center=None,
    pilot_amps_center=None,
):
    """给定当前 I/Q/P 偏压，模拟一次“测量”并提取 12 维特征和 6 维导频指标。

    返回:
        X_sample: [6 个相对功率, 3 个导频频率, 3 个导频幅度]，与数据集一致。
        rel_feats: 长度为 6 的导频一阶/二阶“相对噪底”特征 (dB)。
    """
    if pilot_freqs_center is None:
        pilot_freqs_center = {"I": 0.75e3, "Q": 1e3, "P": 1.25e3}
    if pilot_amps_center is None:
        pilot_amps_center = {"I": 0.02, "Q": 0.02, "P": 0.02}

    xp, use_gpu, cp, mempool, pinned_mempool = init_backend()
    params, total_optical_loss_db, p_in_dbm, ein, responsivity, r_load = build_params()

    # 时间轴
    t = np.arange(0, T_total, 1.0 / Fs)

    # 为本次测量随机导频（与数据集生成时的策略保持一致）
    pilot_freqs_sample = {
        "I": pilot_freqs_center["I"] * np.random.uniform(0.9, 1.1),
        "Q": pilot_freqs_center["Q"] * np.random.uniform(0.9, 1.1),
        "P": pilot_freqs_center["P"] * np.random.uniform(0.9, 1.1),
    }
    pilot_amps_sample = {
        "I": pilot_amps_center["I"] * np.random.uniform(0.8, 1.2),
        "Q": pilot_amps_center["Q"] * np.random.uniform(0.8, 1.2),
        "P": pilot_amps_center["P"] * np.random.uniform(0.8, 1.2),
    }

    V_I_pilot, V_Q_pilot, V_P_pilot = generate_pilot_tones(t, pilot_freqs_sample, pilot_amps_sample)

    # 构造电压波形
    V_I_t_cpu = V_I_bias + V_I_pilot
    V_Q_t_cpu = V_Q_bias + V_Q_pilot
    V_P_t_cpu = V_P_bias + V_P_pilot

    V_I_t = xp.asarray(V_I_t_cpu)
    V_Q_t = xp.asarray(V_Q_t_cpu)
    V_P_t = xp.asarray(V_P_t_cpu)

    Ein_t = xp.ones_like(V_I_t, dtype=xp.complex128) * ein

    # 调制器输出
    E_out_t, P_out_t = run_dpmzm_model(Ein_t, V_I_t, V_Q_t, V_P_t, params, total_optical_loss_db)

    # PD + 噪声
    I_PD_pure, I_PD_noisy, I_AC_t, P_noise_density_dBm_Hz, P_noise_density_W_Hz = photodetect_and_add_noise(
        xp, P_out_t, responsivity, r_load, Fs, use_gpu
    )

    # 频谱
    f, Power_RF_dBm, L = electrical_spectrum(xp, I_AC_t, Fs, r_load)

    if use_gpu and cp is not None:
        f_cpu = cp.asnumpy(f)
        Power_RF_dBm_cpu = cp.asnumpy(Power_RF_dBm)
    else:
        f_cpu = f
        Power_RF_dBm_cpu = Power_RF_dBm

    # 6 个“信号-噪底”特征
    rel_feats = extract_relative_pilot_features(f_cpu, Power_RF_dBm_cpu, pilot_freqs_sample)

    # 6 个导频参数特征
    pilot_param_feats = np.array(
        [
            pilot_freqs_sample["I"],
            pilot_freqs_sample["Q"],
            pilot_freqs_sample["P"],
            pilot_amps_sample["I"],
            pilot_amps_sample["Q"],
            pilot_amps_sample["P"],
        ],
        dtype=float,
    )

    X_sample = np.concatenate([rel_feats, pilot_param_feats], axis=0).astype(np.float32)

    return X_sample, rel_feats


def run_closed_loop(
    n_steps: int = 50,
    init_delta_range: float = 0.5,
    step_gain: float = 0.3,
    model_path: str = "mlp_bias_control_best.pth",
):
    """在仿真环境下演示闭环控制：

    1. 随机一个初始工作点 (相对目标的偏差在 +-init_delta_range V 内)。
    2. 多次：模拟测量 -> 用 MLP 估计当前 ΔV -> 计算控制量 -ΔV -> 更新偏压。
    3. 打印每一步相对目标工作点的偏差，观察是否收敛。
    """
    # 加载模型
    model, device, X_mean, X_std, Y_mean, Y_std = load_trained_model(model_path, device=None)

    # 构造目标工作点（与 generate_dataset 中一致）
    params, *_ = build_params()
    Vpi_I = params["Vpi_I"]
    Vpi_Q = params["Vpi_Q"]
    Vpi_P = params["Vpi_P"]
    BIAS_NULL = 1.0
    BIAS_QUAD = 0.5
    V_I_target = BIAS_NULL * Vpi_I
    V_Q_target = BIAS_NULL * Vpi_Q
    V_P_target = BIAS_QUAD * Vpi_P

    # 随机初始偏差
    dV_init = np.random.uniform(-init_delta_range, init_delta_range, size=3)
    V_I = V_I_target + dV_init[0]
    V_Q = V_Q_target + dV_init[1]
    V_P = V_P_target + dV_init[2]

    print("目标工作点 (V): ")
    print(f"  I: {V_I_target:.3f}, Q: {V_Q_target:.3f}, P: {V_P_target:.3f}")
    print("初始偏压 (V): ")
    print(f"  I: {V_I:.3f}, Q: {V_Q:.3f}, P: {V_P:.3f}")

    # ------------------------------------------------------------
    # 在目标偏压下跑一次导频，得到“理想导频六个指标”作为基准
    # ------------------------------------------------------------
    X_ref, rel_feats_ref = simulate_measurement(V_I_target, V_Q_target, V_P_target)
    rel_feats_ref = np.asarray(rel_feats_ref, dtype=float)
    print("目标偏压下导频六个指标 (dB):", rel_feats_ref)

    # 为方向控制设置每步固定小步长和电压夹紧范围
    base_step = 0.05  # 基础步长 (V)
    k = step_gain      # 步长缩放因子
    step_I = base_step * k
    step_Q = base_step * k
    step_P = base_step * k

    # 电压夹紧范围：相对目标点 ±1 V，可按需要调整
    clamp_margin = 1.0
    V_I_min, V_I_max = V_I_target - clamp_margin, V_I_target + clamp_margin
    V_Q_min, V_Q_max = V_Q_target - clamp_margin, V_Q_target + clamp_margin
    V_P_min, V_P_max = V_P_target - clamp_margin, V_P_target + clamp_margin

    # 使用“导频六个指标”与基准的距离作为代价函数 J
    # 初始化：在初始偏压下测一次 J
    _, rel_feats_init = simulate_measurement(V_I, V_Q, V_P)
    rel_feats_init = np.asarray(rel_feats_init, dtype=float)
    best_cost = float(np.linalg.norm(rel_feats_init - rel_feats_ref))
    best_V_I, best_V_Q, best_V_P = V_I, V_Q, V_P

    for step in range(1, n_steps + 1):
        # 1) 模拟一次测量，得到 12 维特征 X_sample
        X_sample, rel_feats_now = simulate_measurement(V_I, V_Q, V_P)
        rel_feats_now = np.asarray(rel_feats_now, dtype=float)

        # 2) 归一化并送入模型
        x = torch.from_numpy(X_sample).unsqueeze(0).to(device)
        x_norm = (x - X_mean.to(device)) / X_std.to(device)

        with torch.no_grad():
            y_pred_norm = model(x_norm)

        # 3) 反标准化回真实 ΔV 预测（单位 V）
        y_pred = y_pred_norm * Y_std.to(device) + Y_mean.to(device)
        dV_pred = y_pred.squeeze(0).cpu().numpy()  # [dV_I, dV_Q, dV_P]

        # 4) 方向控制：只用符号，步长固定，避免一次走太远
        # 预测的是“当前偏差”，所以控制方向取 -sign(dV_pred)
        dir_I = -np.sign(dV_pred[0]) if dV_pred[0] != 0 else 0.0
        dir_Q = -np.sign(dV_pred[1]) if dV_pred[1] != 0 else 0.0
        dir_P = -np.sign(dV_pred[2]) if dV_pred[2] != 0 else 0.0

        # 尝试步：根据方向加一步并夹紧
        cand_V_I = float(np.clip(V_I + dir_I * step_I, V_I_min, V_I_max))
        cand_V_Q = float(np.clip(V_Q + dir_Q * step_Q, V_Q_min, V_Q_max))
        cand_V_P = float(np.clip(V_P + dir_P * step_P, V_P_min, V_P_max))

        # 在候选偏压下再跑一次导频，计算新的六个指标
        _, rel_feats_cand = simulate_measurement(cand_V_I, cand_V_Q, cand_V_P)
        rel_feats_cand = np.asarray(rel_feats_cand, dtype=float)

        # 以“当前六个指标与参考六个指标的欧氏距离”作为代价函数 J
        cost_now = float(np.linalg.norm(rel_feats_now - rel_feats_ref))
        cost_cand = float(np.linalg.norm(rel_feats_cand - rel_feats_ref))

        # 只在代价变小的时候才接受这一步，否则保持上一步
        accepted = False
        if cost_cand < best_cost:
            V_I, V_Q, V_P = cand_V_I, cand_V_Q, cand_V_P
            best_cost = cost_cand
            best_V_I, best_V_Q, best_V_P = V_I, V_Q, V_P
            accepted = True
        else:
            V_I, V_Q, V_P = best_V_I, best_V_Q, best_V_P
        # 这里的 err_* 和 |err|_2 仅用于仿真打印评估，在真实系统中不可用
        err_I = V_I - V_I_target
        err_Q = V_Q - V_Q_target
        err_P = V_P - V_P_target
        err_norm = float(np.sqrt(err_I**2 + err_Q**2 + err_P**2))

        print(f"Step {step:02d}:")
        print(f"  预测 ΔV (V):  dV_I={dV_pred[0]:+.4f}, dV_Q={dV_pred[1]:+.4f}, dV_P={dV_pred[2]:+.4f}")
        print(f"  候选偏压 (V): I={cand_V_I:.4f}, Q={cand_V_Q:.4f}, P={cand_V_P:.4f}, accepted={accepted}")
        print(f"  导频代价 (J): now={cost_now:.4f}, cand={cost_cand:.4f}, best={best_cost:.4f}")
        print(f"  当前生效偏压 (V): I={V_I:.4f}, Q={V_Q:.4f}, P={V_P:.4f}")
        print(f"  相对目标误差 (V): I={err_I:+.4f}, Q={err_Q:+.4f}, P={err_P:+.4f}, |err|_2={err_norm:.4f}")

    print("闭环仿真结束。")


if __name__ == "__main__":
    # 方向控制 + 固定小步长 + 电压夹紧
    run_closed_loop(n_steps=40, init_delta_range=0.5, step_gain=0.5)
