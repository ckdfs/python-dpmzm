import numpy as np
import matplotlib.pyplot as plt
from plot import plot_electrical_spectrum, plot_optical_spectrum
from dpmzm_nonideal_model import dpmzm_nonideal_model


def init_backend():
    """初始化计算后端(GPU/CPU)，返回 xp, use_gpu, cp(或None), mempool, pinned_mempool。"""
    try:
        import cupy as cp  # type: ignore
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        xp = cp
        use_gpu = True

        dev = cp.cuda.Device()
        # print(f"检测到 GPU: {cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode('utf-8')}")
        return xp, use_gpu, cp, mempool, pinned_mempool
    except ImportError:
        import numpy as cp  # Fallback 仅保证属性存在
        xp = np
        use_gpu = False
        print("未检测到支持 CUDA 的 GPU (CuPy)，将回退到 CPU 模式（速度较慢）。")
        return xp, use_gpu, None, None, None


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


def generate_signals(xp, params, Fs=10e9, T_total=10e-4,
    rf_I=None, rf_Q=None, rf_P=None,
    dither_I=None, dither_Q=None, dither_P=None,
):
    """构造输入信号。"""
    # 时间轴
    t = xp.asarray(np.arange(0, T_total, 1 / Fs))

    # 偏置点系数
    BIAS_MAX = 0.0
    BIAS_QUAD = 0.5
    BIAS_NULL = 1.0

    V_bias_I = BIAS_NULL * params["Vpi_I"]
    V_bias_Q = BIAS_NULL * params["Vpi_Q"]
    V_bias_P = BIAS_QUAD * params["Vpi_P"]

    # 初始化三臂电压为偏置值
    V_I_t = xp.full_like(t, V_bias_I)
    V_Q_t = xp.full_like(t, V_bias_Q)
    V_P_t = xp.full_like(t, V_bias_P)

    # 把 None 统一转成空列表，调用方必须显式传入需要的 RF/dither
    rf_I = rf_I or []
    rf_Q = rf_Q or []
    rf_P = rf_P or []
    dither_I = dither_I or []
    dither_Q = dither_Q or []
    dither_P = dither_P or []

    # 帮助函数: 在某一臂上叠加多路正弦
    def add_tones(V_base, tone_list):
        for tone in tone_list:
            f = tone.get("f", 0.0)
            A = tone.get("A", 0.0)
            if A == 0.0 or f == 0.0:
                continue
            V_base = V_base + A * xp.sin(2 * xp.pi * f * t)
        return V_base

    # 叠加 RF 和 dither
    V_I_t = add_tones(V_I_t, rf_I)
    V_I_t = add_tones(V_I_t, dither_I)

    V_Q_t = add_tones(V_Q_t, rf_Q)
    V_Q_t = add_tones(V_Q_t, dither_Q)

    V_P_t = add_tones(V_P_t, rf_P)
    V_P_t = add_tones(V_P_t, dither_P)

    # 返回一个代表“主 RF 频率”的 f_rf，默认取 I 臂第一个 RF
    if rf_I:
        f_rf_main = rf_I[0]["f"]
    elif rf_Q:
        f_rf_main = rf_Q[0]["f"]
    elif rf_P:
        f_rf_main = rf_P[0]["f"]
    else:
        f_rf_main = 0.0

    return t, V_I_t, V_Q_t, V_P_t, Fs, f_rf_main


def run_dpmzm_model(Ein, V_I_t, V_Q_t, V_P_t, params, total_optical_loss_db):
    """运行 DPMZM 模型并应用物理插损。"""
    E_out_t, P_out_ideal_t = dpmzm_nonideal_model(Ein, V_I_t, V_Q_t, V_P_t, params)
    P_out_t = P_out_ideal_t * (10 ** (-total_optical_loss_db / 10.0))
    return E_out_t, P_out_t


def photodetect_and_add_noise(xp, P_out_t, responsivity, r_load, Fs, use_gpu):
    """PD 探测、计算噪声并生成含噪电流与 AC 分量。"""
    I_PD_pure = responsivity * P_out_t
    I_av = xp.mean(I_PD_pure)

    K = 1.37e-23
    T = 290.0
    B = 1.0
    q = 1.6e-19
    RIN_dB_Hz = -145.0

    if use_gpu:
        import cupy as cp  # type: ignore
        I_av_scalar = float(cp.asnumpy(I_av))
    else:
        I_av_scalar = float(I_av)

    P_thermal_Hz = K * T * B
    P_shot_Hz = 2 * q * I_av_scalar * B * r_load
    P_RIN_Hz = (10 ** (RIN_dB_Hz / 10.0)) * B * r_load * (I_av_scalar ** 2)

    P_noise_density_W_Hz = P_thermal_Hz + P_shot_Hz + P_RIN_Hz
    P_noise_density_dBm_Hz = 10 * np.log10(P_noise_density_W_Hz) + 30.0

    P_noise_total_W = P_noise_density_W_Hz * Fs
    I_noise_rms = np.sqrt(P_noise_total_W / r_load)
    noise_vector = I_noise_rms * xp.random.randn(len(I_PD_pure))

    I_PD_noisy = I_PD_pure + noise_vector
    I_AC_t = I_PD_noisy - xp.mean(I_PD_noisy)

    return I_PD_pure, I_PD_noisy, I_AC_t, P_noise_density_dBm_Hz, P_noise_density_W_Hz


def electrical_spectrum(xp, I_AC_t, Fs, r_load):
    """计算电学频谱和频率轴。"""
    L = len(I_AC_t)
    Y = xp.fft.fft(I_AC_t)

    P2 = xp.abs(Y / L)
    limit = L // 2 + 1
    P1 = P2[0:limit]
    P1[1:-1] *= 2.0

    f = Fs * xp.arange(0, limit) / L
    Power_RF_dBm = 10 * xp.log10((P1 ** 2 / 2.0) * r_load / 1e-3)
    return f, Power_RF_dBm, L


def free_memory(use_gpu, variables_to_del, mempool=None, pinned_mempool=None):
    """删除给定变量并在 GPU 模式下清理显存池。"""
    for v in variables_to_del:
        try:
            del v
        except Exception:
            pass

    if use_gpu and mempool is not None and pinned_mempool is not None:
        import gc

        gc.collect()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()


def run_simulation():
    """一次完整仿真流程：参数设置、信号生成、仿真、绘图和显存释放。"""
    xp, use_gpu, cp, mempool, pinned_mempool = init_backend()
    params, total_optical_loss_db, p_in_dbm, ein, responsivity, r_load = build_params()
    # I 臂: 1 GHz RF + 1.25 kHz dither
    # Q/P 臂: 仅偏置, 不加 RF/dither
    rf_I = [{"f": 1e9, "A": 1.0}]
    dither_I = [{"f": 1.25e3, "A": 0.01}]
    rf_Q = []
    dither_Q = []
    rf_P = []
    dither_P = []

    t, V_I_t, V_Q_t, V_P_t, Fs, f_rf = generate_signals(
        xp,
        params,
        rf_I=rf_I,
        rf_Q=rf_Q,
        rf_P=rf_P,
        dither_I=dither_I,
        dither_Q=dither_Q,
        dither_P=dither_P,
    )
    E_out_t, P_out_t = run_dpmzm_model(ein, V_I_t, V_Q_t, V_P_t, params, total_optical_loss_db)

    P_PD_Input_Watt = xp.mean(P_out_t)
    if use_gpu and cp is not None:
        P_PD_Input_Watt_scalar = float(cp.asnumpy(P_PD_Input_Watt))
    else:
        P_PD_Input_Watt_scalar = float(P_PD_Input_Watt)

    P_PD_Input_dBm = 10 * np.log10(P_PD_Input_Watt_scalar) + 30.0
    print("---------------- 功率分析 ----------------")
    print(f"输入光功率:     {p_in_dbm} dBm")
    print(f"PD输入平均功率: {P_PD_Input_dBm} dBm")

    I_PD_pure, I_PD_noisy, I_AC_t, P_noise_density_dBm_Hz, P_noise_density_W_Hz = photodetect_and_add_noise(
        xp, P_out_t, responsivity, r_load, Fs, use_gpu
    )
    print(f"总噪声密度:     {P_noise_density_dBm_Hz} dBm/Hz")

    f, Power_RF_dBm, L = electrical_spectrum(xp, I_AC_t, Fs, r_load)
    plot_electrical_spectrum(f, Power_RF_dBm, Fs, L, f_rf, P_noise_density_dBm_Hz, use_gpu)
    plot_optical_spectrum(xp, E_out_t, Fs, t, f_rf, use_gpu)

    plt.show()

    free_memory(
        use_gpu,
        [V_I_t, V_Q_t, V_P_t, P_out_t, I_PD_pure, I_PD_noisy, I_AC_t, t, f, Power_RF_dBm],
        mempool,
        pinned_mempool,
    )


if __name__ == "__main__":
    run_simulation()