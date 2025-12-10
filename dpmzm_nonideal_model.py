import torch


def dpmzm_nonideal_model(Ein: torch.Tensor, V_I: torch.Tensor, V_Q: torch.Tensor, V_P: torch.Tensor, params: dict):
    """非理想 DPMZM 模型的 torch 实现。"""

    Vpi_I = params["Vpi_I"]
    Vpi_Q = params["Vpi_Q"]
    Vpi_P = params["Vpi_P"]

    ER_I_lin = 10 ** (params["ER_I_dB"] / 10.0)
    ER_Q_lin = 10 ** (params["ER_Q_dB"] / 10.0)
    ER_P_lin = 10 ** (params["ER_P_dB"] / 10.0)

    device = Ein.device
    delta_I = 1.0 / (torch.sqrt(torch.tensor(ER_I_lin, device=device, dtype=torch.float64)) - 1.0)
    delta_Q = 1.0 / (torch.sqrt(torch.tensor(ER_Q_lin, device=device, dtype=torch.float64)) - 1.0)
    delta_P = 1.0 / (torch.sqrt(torch.tensor(ER_P_lin, device=device, dtype=torch.float64)) - 1.0)

    phi_I = torch.pi * (V_I / Vpi_I)
    phi_Q = torch.pi * (V_Q / Vpi_Q)
    phi_P = torch.pi * (V_P / Vpi_P)

    E_child_in = Ein / 2.0

    # 基于 Li et al. 2018 的非理想项
    E_I_out = E_child_in * (torch.cos(phi_I / 2.0) + delta_I * torch.exp(-1j * phi_I / 2.0))
    E_Q_out = E_child_in * (torch.cos(phi_Q / 2.0) + delta_Q * torch.exp(-1j * phi_Q / 2.0))

    E_out = E_I_out * (1.0 + delta_P) + E_Q_out * torch.exp(1j * phi_P)
    Power_out = torch.abs(E_out) ** 2
    return E_out, Power_out