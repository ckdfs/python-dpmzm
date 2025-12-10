import numpy as np
import torch
from dpmzm_nonideal_model import dpmzm_nonideal_model
from test_dpmzm import build_params, photodetect_and_add_noise, electrical_spectrum


def generate_pilot_tones(t, freqs, amps):
    """在 I/Q/P 三路上生成导频正弦信号。

    freqs: dict, keys in {"I", "Q", "P"}, value = 导频频率 (Hz)
    amps:  dict, keys in {"I", "Q", "P"}, value = 导频幅度 (V)
    返回: V_I_pilot, V_Q_pilot, V_P_pilot
    """
    V_I = np.zeros_like(t)
    V_Q = np.zeros_like(t)
    V_P = np.zeros_like(t)

    if "I" in freqs and freqs["I"] > 0:
        V_I += amps.get("I", 0.0) * np.sin(2 * np.pi * freqs["I"] * t)
    if "Q" in freqs and freqs["Q"] > 0:
        V_Q += amps.get("Q", 0.0) * np.sin(2 * np.pi * freqs["Q"] * t)
    if "P" in freqs and freqs["P"] > 0:
        V_P += amps.get("P", 0.0) * np.sin(2 * np.pi * freqs["P"] * t)

    return V_I, V_Q, V_P


def extract_relative_pilot_features(freqs_fft, spectrum_dBm, pilot_freqs, guard_bins=3):
    """在电学频谱上提取导频一阶/二阶的“相对噪底”特征。

    freqs_fft: 频率轴 (Hz)
    spectrum_dBm: 对应功率谱 (dBm)
    pilot_freqs: dict, {"I": f_I, "Q": f_Q, "P": f_P}

    返回: 长度为 6 的 np.array
        [ΔP(f_I), ΔP(f_Q), ΔP(f_P), ΔP(2f_I), ΔP(2f_Q), ΔP(2f_P)] (dB)，
    其中 ΔP(f) = P_signal(f) - P_noise_local(f)。
    """
    def find_nearest_idx(target_f):
        return int(np.argmin(np.abs(freqs_fft - target_f)))

    def local_noise_floor(idx, width=5):
        n = len(spectrum_dBm)
        left = max(0, idx - guard_bins - width)
        right = min(n, idx + guard_bins + width + 1)
        mask = np.ones(right - left, dtype=bool)
        center_rel = idx - left
        mask[max(0, center_rel - guard_bins):min(right - left, center_rel + guard_bins + 1)] = False
        candidates = spectrum_dBm[left:right][mask]
        if candidates.size == 0:
            return spectrum_dBm[idx]
        return np.mean(candidates)

    f_I = pilot_freqs["I"]
    f_Q = pilot_freqs["Q"]
    f_P = pilot_freqs["P"]

    idx_fI = find_nearest_idx(f_I)
    idx_fQ = find_nearest_idx(f_Q)
    idx_fP = find_nearest_idx(f_P)
    idx_2fI = find_nearest_idx(2 * f_I)
    idx_2fQ = find_nearest_idx(2 * f_Q)
    idx_2fP = find_nearest_idx(2 * f_P)

    signal_idxs = [idx_fI, idx_fQ, idx_fP, idx_2fI, idx_2fQ, idx_2fP]
    rel_feats = []
    for idx in signal_idxs:
        p_sig = spectrum_dBm[idx]
        p_noise = local_noise_floor(idx)
        rel_feats.append(p_sig - p_noise)

    return np.array(rel_feats, dtype=float)


def generate_dataset(
    N_samples=1000,
    Fs=10e6,
    T_total=5e-3,
    pilot_freqs=None,
    pilot_amps=None,
    deltaV_range=0.5,
    save_path="dataset_bias_control.npz",
):
    """生成用于偏置控制的训练数据集。

    输入特征: 三路导频的一阶/二阶六个频点功率 (dBm)
    标签: I/Q/P 三路偏置电压改变值 [dV_I, dV_Q, dV_P] (单位: V)
    """
    if pilot_freqs is None:
        pilot_freqs = {"I": 0.75e3, "Q": 1e3, "P": 1.25e3}
    if pilot_amps is None:
        pilot_amps = {"I": 0.02, "Q": 0.02, "P": 0.02}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params, total_optical_loss_db, p_in_dbm_base, ein_base, responsivity, r_load = build_params()

    # 时间轴（torch，便于直接上 GPU）
    t_torch = torch.arange(0, T_total, 1.0 / Fs, device=device, dtype=torch.float64)

    # 定义一个基准偏置点 (此处使用 build_params 中的 NULL/QUAD 方案)
    BIAS_MAX = 0.0
    BIAS_QUAD = 0.5
    BIAS_NULL = 1.0
    V_bias_I0 = BIAS_NULL * params["Vpi_I"]
    V_bias_Q0 = BIAS_NULL * params["Vpi_Q"]
    V_bias_P0 = BIAS_QUAD * params["Vpi_P"]

    # 特征：6 个相对功率 + 6 个导频参数 (频率、幅度)
    X = np.zeros((N_samples, 12), dtype=float)
    Y = np.zeros((N_samples, 3), dtype=float)

    for n in range(N_samples):
        # 随机采样当前样本的偏置改变量 ΔV_I/Q/P
        dV_I = np.random.uniform(-deltaV_range, deltaV_range)
        dV_Q = np.random.uniform(-deltaV_range, deltaV_range)
        dV_P = np.random.uniform(-deltaV_range, deltaV_range)

        V_I_bias = V_bias_I0 + dV_I
        V_Q_bias = V_bias_Q0 + dV_Q
        V_P_bias = V_bias_P0 + dV_P

        # 为本样本随机导频频率和幅度（在给定中心附近小范围浮动）
        pilot_freqs_sample = {
            "I": pilot_freqs["I"] * np.random.uniform(0.9, 1.1),
            "Q": pilot_freqs["Q"] * np.random.uniform(0.9, 1.1),
            "P": pilot_freqs["P"] * np.random.uniform(0.9, 1.1),
        }
        pilot_amps_sample = {
            "I": pilot_amps["I"] * np.random.uniform(0.8, 1.2),
            "Q": pilot_amps["Q"] * np.random.uniform(0.8, 1.2),
            "P": pilot_amps["P"] * np.random.uniform(0.8, 1.2),
        }

        # 用 PyTorch 在 GPU/CPU 上生成导频并完成全链路计算
        V_I_pilot = pilot_amps_sample["I"] * torch.sin(2 * torch.pi * t_torch * pilot_freqs_sample["I"])
        V_Q_pilot = pilot_amps_sample["Q"] * torch.sin(2 * torch.pi * t_torch * pilot_freqs_sample["Q"])
        V_P_pilot = pilot_amps_sample["P"] * torch.sin(2 * torch.pi * t_torch * pilot_freqs_sample["P"])

        V_I_t = torch.as_tensor(V_I_bias, device=device) + V_I_pilot
        V_Q_t = torch.as_tensor(V_Q_bias, device=device) + V_Q_pilot
        V_P_t = torch.as_tensor(V_P_bias, device=device) + V_P_pilot

        p_in_dbm = p_in_dbm_base + np.random.uniform(-3.0, 3.0)
        ein = np.sqrt(10 ** ((p_in_dbm - 30.0) / 10.0))
        Ein_t = torch.ones_like(V_I_t, dtype=torch.complex128) * ein

        E_out_t, P_out_ideal = dpmzm_nonideal_model(Ein_t, V_I_t, V_Q_t, V_P_t, params)
        P_out_t = P_out_ideal * (10 ** (-total_optical_loss_db / 10.0))

        # PD + 噪声 + 频谱（torch）
        _, _, I_AC_t, P_noise_density_dBm_Hz, P_noise_density_W_Hz = photodetect_and_add_noise(
            P_out_t, responsivity, r_load, Fs
        )
        f_t, Power_RF_dBm_t, L = electrical_spectrum(I_AC_t, Fs, r_load)

        # 搬回 CPU 做特征
        f_cpu = f_t.cpu().numpy()
        Power_RF_dBm_cpu = Power_RF_dBm_t.cpu().numpy()

        # 提取 6 个“信号-本地噪底”特征
        rel_feats = extract_relative_pilot_features(f_cpu, Power_RF_dBm_cpu, pilot_freqs_sample)

        # 构造导频参数特征（频率、幅度），统一按 Hz 和 V 存储
        pilot_param_feats = np.array([
            pilot_freqs_sample["I"],
            pilot_freqs_sample["Q"],
            pilot_freqs_sample["P"],
            pilot_amps_sample["I"],
            pilot_amps_sample["Q"],
            pilot_amps_sample["P"],
        ], dtype=float)

        X[n, :] = np.concatenate([rel_feats, pilot_param_feats], axis=0)

        # 标签为 ΔV
        Y[n, :] = np.array([dV_I, dV_Q, dV_P], dtype=float)

        if (n + 1) % max(1, N_samples // 10) == 0:
            print(f"生成样本 {n + 1}/{N_samples}")

    # 保存数据集（同时保存“中心导频设置”，方便记录）
    np.savez(
        save_path,
        X=X,
        Y=Y,
        pilot_freqs_center=pilot_freqs,
        pilot_amps_center=pilot_amps,
        Fs=Fs,
        T_total=T_total,
        deltaV_range=deltaV_range,
    )
    print(f"数据集已保存到 {save_path}")


if __name__ == "__main__":
    generate_dataset(N_samples=100000, save_path="dataset_bias_control.npz")
