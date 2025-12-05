import numpy as np
import matplotlib.pyplot as plt


def _annotate_rf_orders(f_cpu, p_cpu, f_rf, orders=(1, 2), color="orange"):
    """在电学频谱上标注主 RF 的若干阶谐波功率。

    f_cpu: 频率轴 (Hz)
    p_cpu: 对应功率 (dBm)
    f_rf:  主 RF 频率 (Hz)
    orders: 要标注的阶数，例如 (1, 2) 表示 1f 和 2f
    """
    if f_rf is None or f_rf <= 0:
        return

    # 如果频率数组太短或全为零，直接返回
    if len(f_cpu) == 0 or np.all(~np.isfinite(p_cpu)):
        return

    for n in orders:
        target = n * f_rf
        # 在 target 附近 ±5% * f_rf 的窗口中寻找峰值
        delta = 0.05 * f_rf
        mask = (f_cpu > target - delta) & (f_cpu < target + delta)
        if not np.any(mask):
            continue

        local_f = f_cpu[mask]
        local_p = p_cpu[mask]
        idx = np.argmax(local_p)
        peak_f = local_f[idx]
        peak_p = local_p[idx]

        x_mhz = peak_f / 1e6
        plt.scatter(x_mhz, peak_p, color=color, s=20, zorder=3)
        plt.text(
            x_mhz,
            peak_p + 2,
            f"{n}×RF\n{peak_p:.1f} dBm",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_electrical_spectrum(f, Power_RF_dBm, Fs, L, f_rf,
                             P_noise_density_dBm_Hz, use_gpu):
    """绘制电学频谱图。"""
    if use_gpu:
        import cupy as cp  # type: ignore
        f_cpu = cp.asnumpy(f)
        p_cpu = cp.asnumpy(Power_RF_dBm)
    else:
        f_cpu = f
        p_cpu = Power_RF_dBm

    RBW_Sim_Hz = Fs / L
    Calculated_Floor_dBm = P_noise_density_dBm_Hz + 10 * np.log10(RBW_Sim_Hz)

    f_show_max = f_rf * 5
    idx_show = f_cpu <= f_show_max

    f_plot = f_cpu[idx_show]
    p_plot = p_cpu[idx_show]

    plt.figure(figsize=(10, 5))
    plt.plot(f_plot / 1e6, p_plot, color="#3366cc", linewidth=1)
    plt.axhline(
        y=Calculated_Floor_dBm,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=f"Noise Floor @ RBW={RBW_Sim_Hz/1e3:.1f}kHz",
    )

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("RF Power (dBm)")
    plt.title("Electrical Spectrum at PD Output")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim(bottom=-160, top=np.max(p_plot) + 10)

    # 在主 RF 的 1 阶、2 阶附近标注功率
    _annotate_rf_orders(f_plot, p_plot, f_rf, orders=(1, 2))

    plt.legend()
    plt.tight_layout()


def _annotate_optical_orders(f_plot_opt, p_plot_opt, f_rf, max_order=2):
    """在光学频谱上标注载波和 ±若干阶边带功率。

    f_plot_opt: 相对载波频率轴 (GHz)
    p_plot_opt: 功率 (dBm)
    f_rf: 主 RF 频率 (Hz)
    """
    if f_rf is None or f_rf <= 0:
        return

    f_rf_ghz = f_rf / 1e9

    def find_peak(center_ghz, span_factor=0.2):
        # 在 center_ghz ± span_factor*f_rf_ghz 的范围内找峰
        delta = span_factor * f_rf_ghz
        mask = (f_plot_opt > center_ghz - delta) & (f_plot_opt < center_ghz + delta)
        if not np.any(mask):
            return None, None
        local_f = f_plot_opt[mask]
        local_p = p_plot_opt[mask]
        idx = np.argmax(local_p)
        return local_f[idx], local_p[idx]

    # 载波 (0 GHz)
    c_f, c_p = find_peak(0.0, span_factor=0.1)
    if c_f is not None:
        plt.scatter(c_f, c_p, color="black", s=25, zorder=3)
        plt.text(
            c_f,
            c_p + 1.5,
            f"Carrier\n{c_p:.1f} dBm",
            ha="center",
            fontsize=8,
        )

    # 正负阶边带
    for n in range(1, max_order + 1):
        # +n 阶
        center_pos = n * f_rf_ghz
        pf, pp = find_peak(center_pos)
        if pf is not None:
            plt.scatter(pf, pp, color="blue", s=20, zorder=3)
            plt.text(
                pf,
                pp + 1.5,
                f"+{n}th\n{pp:.1f} dBm",
                ha="center",
                fontsize=8,
            )

        # -n 阶
        center_neg = -n * f_rf_ghz
        nf, np_ = find_peak(center_neg)
        if nf is not None:
            plt.scatter(nf, np_, color="blue", s=20, zorder=3)
            plt.text(
                nf,
                np_ + 1.5,
                f"-{n}th\n{np_:.1f} dBm",
                ha="center",
                fontsize=8,
            )


def plot_optical_spectrum(xp, E_out_t, Fs, t, f_rf, use_gpu):
    """绘制光学频谱图。"""
    E_spec_gpu = xp.fft.fft(E_out_t)
    E_spec_gpu = xp.fft.fftshift(E_spec_gpu)

    P_spec_gpu = xp.abs(E_spec_gpu) ** 2
    P_time_avg = xp.mean(xp.abs(E_out_t) ** 2)

    scale_factor = P_time_avg / xp.sum(P_spec_gpu)
    P_spec_gpu = P_spec_gpu * scale_factor

    P_spec_dBm_gpu = 10 * xp.log10(P_spec_gpu * 1000 + 1e-20)

    L = len(t)
    f_opt_gpu = Fs * xp.arange(-L // 2, L // 2) / L

    if use_gpu:
        import cupy as cp  # type: ignore
        f_opt_cpu = cp.asnumpy(f_opt_gpu)
        P_spec_dBm_cpu = cp.asnumpy(P_spec_dBm_gpu)
    else:
        f_opt_cpu = f_opt_gpu
        P_spec_dBm_cpu = P_spec_dBm_gpu

    disp_span = 2.5 * f_rf
    idx_opt = (f_opt_cpu >= -disp_span) & (f_opt_cpu <= disp_span)

    f_plot_opt = f_opt_cpu[idx_opt] / 1e9
    p_plot_opt = P_spec_dBm_cpu[idx_opt]

    plt.figure(figsize=(10, 6))
    plt.plot(f_plot_opt, p_plot_opt, color="#cc3333", linewidth=1)

    # 标注载波及 ±1、±2 阶边带功率
    _annotate_optical_orders(f_plot_opt, p_plot_opt, f_rf, max_order=2)

    plt.xlabel("Frequency Relative to Carrier (GHz)")
    plt.ylabel("Optical Power (dBm)")
    plt.title("Optical Spectrum (OSA View)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim(bottom=-100)
    plt.tight_layout()