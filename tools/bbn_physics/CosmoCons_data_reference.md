# CosmoCons.dat 数据格式参考

数据来源：`/mnt/d/User/Desktop/Project/BBNHHe/CosmoCons.dat`
生成程序：`bbn_kmmHHe.f`（Kawano BBN code，经 kmm 修改）
理论参考：Wagoner, Fowler & Hoyle 1967, Ap.J. 148

## 列定义

| 列号 | 列名 | 单位 | 说明 | 对应代码变量 |
|------|-------|------|------|-------------|
| 1 | Temp | MeV | 宇宙温度 | `t9out(j)` (内部 t9 以 $10^9$ K 为单位, $1\text{ MeV} = 11.605 \times 10^9\text{ K}$) |
| 2 | time | s | 宇宙年龄（从大爆炸起算） | `tout(j)` |
| 3 | rhog | $\text{g/cm}^3$ | 光子能量密度（等价质量密度 $\rho/c^2$） | `thmout(j,1)` = `thm(1)` |
| 4 | rhoe | $\text{g/cm}^3$ | 电子+正电子能量密度 | `thmout(j,2)` = `thm(4)` |
| 5 | rhone | $\text{g/cm}^3$ | 中微子能量密度（所有种类之和） | `thmout(j,3)` = `thm(8)` |
| 6 | rhob | $\text{g/cm}^3$ | 重子质量密度 | `thmout(j,4)` = `thm(9)` |
| 7 | phie | 无量纲 | 电子化学势 $\phi_e = \mu_e / (k_B T)$ | `thmout(j,5)` = `phie` |
| 8 | rhotot | $\text{g/cm}^3$ | 总能量密度 $= \rho_\gamma + \rho_{e^\pm} + \rho_\nu + \rho_b$ | `thmout(j,6)` = `thm(10)` |
| 9 | H | $\text{s}^{-1}$ | Hubble 膨胀速率 $H = \sqrt{8\pi G \rho_\text{tot}/3}$ | `hubout(j)` = `hubcst` |
| 10 | Ne | $\text{cm}^{-3}$ | 电子+正电子数密度 | — |
| 11 | Ng | $\text{cm}^{-3}$ | 光子数密度 | — |
| 12 | nLamdg | cm ? | 物理意义待确定 | — |
| 13 | LamdD | cm ? | 可能与 Debye 屏蔽长度有关 | — |
| 14 | c/H | cm | Hubble 半径 $= c / H$ | — |

## 关键公式（源自 bbn_kmmHHe.f therm 子程序）

### 光子能量密度 (thm(1))
$$\rho_\gamma = 8.418 \times t_9^4 \quad [\text{g/cm}^3]$$

其中 $t_9 = T / (10^9\text{ K})$。系数 8.418 来自 Wagoner 1967, Eq. A2:
$$\rho_\gamma = \frac{\pi^2}{15} \frac{(k_B T)^4}{(\hbar c)^3} \frac{1}{c^2} = a_\text{rad} T^4 / c^2$$

### 电子+正电子能量密度 (thm(4))
$$\rho_{e^\pm} = 3206 \sum_{n=1}^{5} (-1)^{n+1} b_n(z) \cosh(n\phi_e)$$

其中 $z = m_e c^2 / (k_B T) = 5.930 / t_9$，$b_n$ 为修正 Bessel 函数相关量。
参考：Fowler & Hoyle 1964, Eq. B44。

### 中微子能量密度 (thm(8))
非简并情况：
$$\rho_\nu = N_\nu \times 7.366 \times t_9^4 \quad [\text{g/cm}^3]$$

$N_\nu$ 为中微子种类数。简并情况需逐种计算。
参考：Wagoner 1967, Eq. A4。注意 7.366 是单种中微子的系数（14.73 的一半对应两种）。

### 重子密度 (thm(9))
$$\rho_b = \rho_{b,0} \times r_{nb}$$

初始重子密度 $\rho_{b,0} = h_v \times t_9^3$，其中 $h_v = 3.3683 \times 10^4 \times \eta \times 2.75$。

$\eta$ 为重子-光子比。2.75 因子对应初始与最终 $\eta$ 值的 11/4 倍差异。
参考：Wagoner 1969, Eq. 4。

### 电子化学势 (phie)
$$\phi_e = h_v \times (1.784 \times 10^{-5} \times Y_p) / \left[\frac{1}{2} z^3 (b_{l,1} - 2b_{l,2} + 3b_{l,3} - 4b_{l,4} + 5b_{l,5})\right]$$

参考：Kawano 1992, FERMILAB-PUB-92/04-A, Eq. D.2。

### Hubble 参数
$$H = \sqrt{\frac{8\pi G \rho_\text{tot}}{3}}$$

### 验证
以第一行数据为例（Temp = 8.617 MeV, 即 $t_9 = 100.0$）：
- $\rho_\gamma = 8.418 \times 100^4 = 8.418 \times 10^8$ ✓
- $H = 50.29\text{ s}^{-1}$ → $c/H = 3\times10^{10} / 50.29 = 5.967\times10^8\text{ cm}$ ≈ $5.962\times10^8$ ✓

## 数值范围概览

| 量 | T = 8.617 MeV | T = 0.84 MeV | T = 0.084 MeV |
|----|---------------|--------------|---------------|
| time | 0.011 s | 176 s | 384 s |
| $\rho_\gamma$ | $8.4\times10^8$ | 7.78 | 1.81 |
| $\rho_{e^\pm}$ | $1.5\times10^9$ | 1.10 | 0.038 |
| $\rho_\nu$ | $2.2\times10^9$ | 6.16 | 1.26 |
| $\rho_b$ | 56.7 | $2.2\times10^{-5}$ | $6.6\times10^{-6}$ |
| $H$ | 50.3 s⁻¹ | $2.9\times10^{-3}$ | $1.3\times10^{-3}$ |
| $c/H$ | $6.0\times10^8$ cm | $1.0\times10^{13}$ | $2.3\times10^{13}$ |

## BBN 相关物理时期

- **T > 0.8 MeV**: $e^\pm$ 对湮灭前，$\rho_{e^\pm}$ 主导或与 $\rho_\gamma$ 相当
- **T ~ 0.5-1 MeV**: 弱反应冻结（中子-质子比冻结）
- **T ~ 0.1 MeV**: D 核合成开始（deuterium bottleneck 打开）
- **T ~ 0.03 MeV**: BBN 核反应基本结束，He-4 丰度冻结

## 温度转换

$$T\ [\text{MeV}] = t_9 / 11.605$$
$$T\ [\text{K}] = t_9 \times 10^9$$
$$1\ \text{MeV}/k_B = 1.1605 \times 10^{10}\text{ K}$$

## 代码参考

- 主程序: `bbn_kmmHHe.f` — Kawano BBN code（H, He 核素）
- 热力学: `therm` 子程序 (line ~2777) — 计算所有热力学量
- 能量损失: `Kenerloss.f` — 读取 CosmoCons 数据计算核子能量损失
- 能量损失物理: `Kenergy_losses.f` — 核子/电子能量损失截面库
