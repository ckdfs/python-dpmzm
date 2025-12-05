import numpy as np

def dpmzm_nonideal_model(Ein, V_I, V_Q, V_P, params):
    """
    DPMZM_NONIDEAL_MODEL 基于 Li 等人 (2018) 论文的非理想 DPMZM 数学模型
    
    说明: 该模型被设计为"0dB 固有插损"模型。
    物理插损 (Insertion Loss) 应在调用该函数后在主脚本中单独计算。
    
    参考文献: Li et al., "Arbitrary Bias Point Control...", JLT 2018
    """
    
    # --- 自动检测计算后端 (NumPy vs CuPy) ---
    # 根据输入数据 Ein 的类型决定使用哪个库进行计算
    try:
        import cupy as cp
        xp = cp.get_array_module(Ein)
    except ImportError:
        xp = np
        
    # %% 1. 提取参数
    Vpi_I = params['Vpi_I']
    Vpi_Q = params['Vpi_Q']
    Vpi_P = params['Vpi_P']
    
    # 将消光比从 dB 转换为线性值
    # MATLAB: 10^(x/10) -> Python: 10**(x/10.0)
    ER_I_lin = 10**(params['ER_I_dB'] / 10.0)
    ER_Q_lin = 10**(params['ER_Q_dB'] / 10.0)
    ER_P_lin = 10**(params['ER_P_dB'] / 10.0)
    
    # %% 2. 计算非理想因子 delta 
    # MATLAB: 1 / (sqrt(ER) - 1)
    delta_I = 1.0 / (xp.sqrt(ER_I_lin) - 1.0)
    delta_Q = 1.0 / (xp.sqrt(ER_Q_lin) - 1.0)
    delta_P = 1.0 / (xp.sqrt(ER_P_lin) - 1.0)
    
    # %% 3. 计算相位延迟 
    # MATLAB: pi -> xp.pi
    phi_I = xp.pi * (V_I / Vpi_I)
    phi_Q = xp.pi * (V_Q / Vpi_Q)
    phi_P = xp.pi * (V_P / Vpi_P)
    
    # %% 4. 计算子调制器输出
    # [Refinement]: 使用 sqrt(4) 是为了功率归一化。
    # 解释: 实际物理分光是 1:2 (Power/2 => Amp/sqrt(2))。
    # 但后续的数学加法 (E_out = E_I + E_Q) 会导致最大振幅变为 2倍。
    # 为了让模型输出的最大光功率等于输入光功率 (即 0dB 固有损耗)，
    # 这里将单臂幅度设为 1/2 (即 sqrt(4))，这样 1/2 + 1/2 = 1。
    E_child_in = Ein / 2.0
    
    # MZM I 输出
    E_I_out = E_child_in * (xp.cos(phi_I / 2.0) + delta_I * xp.exp(-1j * phi_I / 2.0))
    
    # MZM Q 输出
    E_Q_out = E_child_in * (xp.cos(phi_Q / 2.0) + delta_Q * xp.exp(-1j * phi_Q / 2.0))
    
    # %% 5. 计算父级调制器总输出
    # 这里的直接相加模拟了合束器，但在数学上没有除以 sqrt(2)。
    # 结合前面 E_child_in 的设置，整体增益为 1 (0dB)。
    E_out = E_I_out * (1.0 + delta_P) + E_Q_out * xp.exp(1j * phi_P)
    
    # 计算输出功率
    Power_out = xp.abs(E_out)**2
    
    return E_out, Power_out