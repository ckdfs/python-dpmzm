import numpy as np
import torch
import matplotlib.pyplot as plt
from plot import plot_electrical_spectrum, plot_optical_spectrum
from dpmzm_nonideal_model import dpmzm_nonideal_model


def build_params():
    """构造调制器与系统参数。"""
    params = {
        "Vpi_I": 5.0,
        "Vpi_Q": 5.0,
        "Vpi_P": 5.0,
        "ER_I_dB": 30.0,
        "ER_Q_dB": 30.0,
        "ER_P_dB": 30.0,
    }

    loss_insertion_db = 6.0
    loss_tap_db = 0.0
    total_optical_loss_db = loss_insertion_db + loss_tap_db

    p_in_dbm = 20.0
    ein = np.sqrt(10 ** ((p_in_dbm - 30.0) / 10.0))
    responsivity = 0.786
    r_load = 50.0

    return params, total_optical_loss_db, p_in_dbm, ein, responsivity, r_load


def generate_signals(params, Fs=10e9, T_total=10e-4,
    rf_I=None, rf_Q=None, rf_P=None,
    dither_I=None, dither_Q=None, dither_P=None,
    device=None):
    """构造输入信号。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.arange(0, T_total, 1 / Fs, device=device, dtype=torch.float64)

    BIAS_MAX = 0.0
    BIAS_QUAD = 0.5
    BIAS_NULL = 1.0

    V_bias_I = BIAS_NULL * params["Vpi_I"]
    V_bias_Q = BIAS_NULL * params["Vpi_Q"]
    V_bias_P = BIAS_QUAD * params["Vpi_P"]

    V_I_t = torch.full_like(t, V_bias_I)
    V_Q_t = torch.full_like(t, V_bias_Q)
    V_P_t = torch.full_like(t, V_bias_P)

    rf_I = rf_I or []
    rf_Q = rf_Q or []
    rf_P = rf_P or []
    dither_I = dither_I or []
    dither_Q = dither_Q or []
    dither_P = dither_P or []

    def add_tones(V_base, tone_list):
        for tone in tone_list:
            f = tone.get("f", 0.0)
            A = tone.get("A", 0.0)
            if A == 0.0 or f == 0.0:
                continue
            V_base = V_base + A * torch.sin(2 * torch.pi * f * t)
        return V_base

    V_I_t = add_tones(V_I_t, rf_I)
    V_I_t = add_tones(V_I_t, dither_I)

    V_Q_t = add_tones(V_Q_t, rf_Q)
    V_Q_t = add_tones(V_Q_t, dither_Q)

    V_P_t = add_tones(V_P_t, rf_P)
    V_P_t = add_tones(V_P_t, dither_P)

    if rf_I:
        f_rf_main = rf_I[0]["f"]
    elif rf_Q:
        f_rf_main = rf_Q[0]["f"]
    elif rf_P:
        f_rf_main = rf_P[0]["f"]
    else:
        f_rf_main = 0.0

    return t, V_I_t, V_Q_t, V_P_t, Fs, f_rf_main


def photodetect_and_add_noise(P_out_t: torch.Tensor, responsivity, r_load, Fs):
    """PD 探测、计算噪声并生成含噪电流与 AC 分量。"""
    device = P_out_t.device
    dtype = P_out_t.dtype

    I_PD_pure = responsivity * P_out_t
    I_av = torch.mean(I_PD_pure)

    K = torch.tensor(1.37e-23, device=device, dtype=dtype)
    T_const = torch.tensor(290.0, device=device, dtype=dtype)
    B = torch.tensor(1.0, device=device, dtype=dtype)
    q = torch.tensor(1.6e-19, device=device, dtype=dtype)
    RIN_dB_Hz = torch.tensor(-145.0, device=device, dtype=dtype)

    base10 = torch.tensor(10.0, device=device, dtype=dtype)

    P_thermal_Hz = K * T_const * B
    P_shot_Hz = 2 * q * I_av * B * r_load
    P_RIN_Hz = torch.pow(base10, RIN_dB_Hz / 10.0) * B * r_load * (I_av ** 2)

    P_noise_density_W_Hz = P_thermal_Hz + P_shot_Hz + P_RIN_Hz
    P_noise_density_dBm_Hz = 10 * torch.log10(P_noise_density_W_Hz) + 30.0

    P_noise_total_W = P_noise_density_W_Hz * Fs
    I_noise_rms = torch.sqrt(P_noise_total_W / r_load)
    noise_vector = I_noise_rms * torch.randn(P_out_t.shape, device=device, dtype=dtype)

    I_PD_noisy = I_PD_pure + noise_vector
    I_AC_t = I_PD_noisy - torch.mean(I_PD_noisy)

    return I_PD_pure, I_PD_noisy, I_AC_t, P_noise_density_dBm_Hz, P_noise_density_W_Hz


def electrical_spectrum(I_AC_t: torch.Tensor, Fs, r_load):
    """计算电学频谱和频率轴。"""
    L = I_AC_t.numel()
    Y = torch.fft.fft(I_AC_t)

    P2 = torch.abs(Y / L)
    limit = L // 2 + 1
    P1 = P2[0:limit]
    if limit > 2:
        P1[1:-1] *= 2.0

    f = Fs * torch.arange(0, limit, device=I_AC_t.device, dtype=torch.float64) / L
    Power_RF_dBm = 10 * torch.log10((P1 ** 2 / 2.0) * r_load / 1e-3)
    return f, Power_RF_dBm, L


def run_simulation():
    """一次完整仿真流程。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params, total_optical_loss_db, p_in_dbm, ein, responsivity, r_load = build_params()

    Fs = 10e9
    T_total = 10e-4

    rf_I = [{"f": 1e9, "A": 1.0}]
    dither_I = [{"f": 1.25e3, "A": 0.01}]
    rf_Q = []
    dither_Q = []
    rf_P = []
    dither_P = []

    t, V_I_t, V_Q_t, V_P_t, Fs, f_rf = generate_signals(
        params,
        Fs=Fs,
        T_total=T_total,
        rf_I=rf_I,
        rf_Q=rf_Q,
        rf_P=rf_P,
        dither_I=dither_I,
        dither_Q=dither_Q,
        dither_P=dither_P,
        device=device,
    )

    Ein_t = torch.ones_like(V_I_t, dtype=torch.complex128) * ein
    E_out_t, P_out_ideal_t = dpmzm_nonideal_model(Ein_t, V_I_t, V_Q_t, V_P_t, params)
    P_out_t = P_out_ideal_t * (10 ** (-total_optical_loss_db / 10.0))

    P_PD_Input_Watt = torch.mean(P_out_t)
    P_PD_Input_Watt_scalar = float(P_PD_Input_Watt.detach().cpu())
    P_PD_Input_dBm = 10 * np.log10(P_PD_Input_Watt_scalar) + 30.0

    print("---------------- 功率分析 ----------------")
    print(f"输入光功率:     {p_in_dbm} dBm")
    print(f"PD输入平均功率: {P_PD_Input_dBm} dBm")

    I_PD_pure, I_PD_noisy, I_AC_t, P_noise_density_dBm_Hz, P_noise_density_W_Hz = photodetect_and_add_noise(
        P_out_t, responsivity, r_load, Fs
    )
    print(f"总噪声密度:     {float(P_noise_density_dBm_Hz.cpu()):.6f} dBm/Hz")

    f, Power_RF_dBm, L = electrical_spectrum(I_AC_t, Fs, r_load)

    f_cpu = f.detach().cpu().numpy()
    Power_RF_dBm_cpu = Power_RF_dBm.detach().cpu().numpy()
    t_cpu = t.detach().cpu().numpy()
    E_out_cpu = E_out_t.detach().cpu().numpy()

    plot_electrical_spectrum(f_cpu, Power_RF_dBm_cpu, Fs, L, f_rf, float(P_noise_density_dBm_Hz.cpu()), use_gpu=torch.cuda.is_available())
    plot_optical_spectrum(np, E_out_cpu, Fs, t_cpu, f_rf, use_gpu=torch.cuda.is_available())

    plt.show()


if __name__ == "__main__":
    run_simulation()