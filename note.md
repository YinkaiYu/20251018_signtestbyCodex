我们正在研究 world line QMC 方法的费米子符号问题。

为简单起见，我们先考虑正方晶格上的自由费米子
（周期性边界条件，尺寸$L\times L$）：
$$
H=-t\sum_{\braket{ij}}
\left(c_{i\uparrow}^\dagger c_{j\uparrow} + c_{i\downarrow}^\dagger c_{j\downarrow}\right)
=\sum_{k\sigma}\varepsilon_k c_{k\sigma}^\dagger c_{k\sigma}
$$
其中 $\varepsilon_k=-2t(\cos k_x + \cos k_y)$.

配分函数写为
$$
Z=\text{tr}\left( e^{-\beta H} \right)=
\frac{1}{N_{\uparrow}!}\frac{1}{N_{\downarrow}!}
\sum_{\{k_{\uparrow}^{(m)}\},P_{\uparrow}}\sum_{\{k_{\downarrow}^{(n)}\},P_{\downarrow}}
\text{sgn}(P_{\uparrow})\text{sgn}(P_{\downarrow})
\prod_{m=1}^{N_{\uparrow}}\bra{k_{\uparrow}^{(P_{\uparrow}m)}}e^{-\beta H}\ket{k_{\uparrow}^{(m)}}
\prod_{n=1}^{N_{\downarrow}}\bra{k_{\downarrow}^{(P_{\downarrow}n)}}e^{-\beta H}\ket{k_{\downarrow}^{(n)}}
$$
其中 $N_{\uparrow}=N_{\downarrow}=L^2/2$，表示费米子数半填充。
$m,n$ 是费米子的索引，$P_{\uparrow},P_{\downarrow}$ 是全同费米子的 permutation。由于粒子数只考虑了半填充，

蒙卡采样权重正比于
$$
w\left(\{k_{\uparrow}^{(m)}\},P_{\uparrow},\{k_{\downarrow}^{(n)}\},P_{\downarrow}\right)\propto
\text{sgn}(P_{\uparrow})\text{sgn}(P_{\downarrow})
\prod_{m=1}^{N_{\uparrow}}\bra{k_{\uparrow}^{(P_{\uparrow}m)}}e^{-\beta H}\ket{k_{\uparrow}^{(m)}}
\prod_{n=1}^{N_{\downarrow}}\bra{k_{\downarrow}^{(P_{\downarrow}n)}}e^{-\beta H}\ket{k_{\downarrow}^{(n)}}\\
=\text{sgn}(P_{\uparrow})\text{sgn}(P_{\downarrow})
\prod_{m=1}^{N_{\uparrow}} e^{-\beta \varepsilon_k} \delta_{P_{\uparrow}m,m} 
\prod_{n=1}^{N_{\downarrow}} e^{-\beta \varepsilon_k} \delta_{P_{\downarrow}n,n} 
$$
这个权重是半正定的，且只有当 identity permutation 时才不为零。

接下来考虑添加随机时空辅助场（实空间随机的Zeeman场）的哈密顿量，这时要考虑Trotter分解（分为 $L_\tau=\beta/\Delta\tau$ 个切片，用 $l$ 来标记第几个切片）. 
记 $K_\sigma=\{k_{l\sigma}^{(n)}\}$ 表示动量空间世界线，$\sigma=\uparrow,\downarrow$.

配分函数写为（Trotter 分解，permutation 仅出现在虚时周期性边界条件处）：
$$
Z=\text{tr}\left( e^{-\beta H} \right)=
\frac{1}{N_{\uparrow}!}\frac{1}{N_{\downarrow}!}
\sum_{K_{\uparrow},P_{\uparrow}}\sum_{K_{\downarrow},P_{\downarrow}}
\text{sgn}(P_{\uparrow})\text{sgn}(P_{\downarrow})
\prod_{\sigma=\uparrow,\downarrow}
\prod_{n=1}^{N_{\sigma}}
\prod_{l=0}^{L_\tau-1}
\bra{k_{l+1,\sigma}^{(n)}}T_{l,\sigma}\ket{k_{l,\sigma}^{(n)}},\quad
\text{其中 }k_{L_\tau,\sigma}^{(n)}\equiv k_{0,\sigma}^{(P_{\sigma}n)}.
$$
其中
$$
\bra{k'}T_{l,\sigma}\ket{k}=\bra{k'}e^{-\frac{\Delta\tau}{2}K}\,e^{-\Delta\tau V_\ell}\,e^{-\frac{\Delta\tau}{2}K}\ket{k}=e^{-\frac{\Delta\tau}{2}(\varepsilon_{k'}+\varepsilon_k)}\frac{W_{l,\sigma}(k-k')}{V}
$$
$$
W_{l,\uparrow}(q)=\sum_{i} e^{i q r_i} e^{\lambda s_{il}},\quad
W_{l,\downarrow}(q)=\sum_{i} e^{i q r_i} e^{-\lambda s_{il}}
$$
$$
\lambda=\cosh^{-1}(e^{\Delta\tau U/2}), \quad V=L^2
$$
时空随机辅助场 $s_{il}=\pm 1$，$i$ 为空间索引，$l$ 为时间索引。
$W_{l,\sigma}(q)$ 可以用FFT提前计算好。

蒙卡采样权重为（注意 $P_\sigma$ 仅通过边界 $l=L_\tau-1\to 0$ 的链接进入）
$$
w\left(K_{\uparrow},P_{\uparrow},K_{\downarrow},P_{\downarrow}\right)\propto
\text{sgn}(P_{\uparrow})\text{sgn}(P_{\downarrow})
\prod_{\sigma=\uparrow,\downarrow}
\prod_{n=1}^{N_{\sigma}}
\Bigg[\prod_{l=0}^{L_\tau-2}\bra{k_{l+1,\sigma}^{(n)}}T_{l,\sigma}\ket{k_{l,\sigma}^{(n)}}\Bigg]
\times \bra{k_{0,\sigma}^{(P_{\sigma}n)}}T_{L_\tau-1,\sigma}\ket{k_{L_\tau-1,\sigma}^{(n)}}
$$

蒙卡更新分别对 $K_{\sigma}$ 和 $P_{\sigma}$ 进行。注意：辅助场 $\{s_{il}\}$ 在模拟开始时一次性生成并固定，整个采样过程中不再更新（不采样辅助场），且全程不使用行列式；我们仅对动量空间世界线与 permutation 采样，观测量是本方法的平均符号/相位因子。

为便于记号，定义单片上的转移矩阵元
$$
M_{l,\sigma}(k'\leftarrow k)
=\bra{k'}T_{l,\sigma}\ket{k}
=e^{-\frac{\Delta\tau}{2}(\varepsilon_{k'}+\varepsilon_k)}\frac{W_{l,\sigma}(k-k')}{V}.
$$
对任意配置 $X\equiv (K_{\uparrow},P_{\uparrow};K_{\downarrow},P_{\downarrow})$，
其（可能为复数的）权重为（$P$ 只出现在边界片 $l=L_\tau-1$）：
$$
w(X)=\text{sgn}(P_{\uparrow})\,\text{sgn}(P_{\downarrow})\,\prod_{\sigma}\prod_{n=1}^{N_\sigma}\Bigg[\prod_{l=0}^{L_\tau-2} M_{l,\sigma}\big(k_{l+1,\sigma}^{(n)}\leftarrow k_{l,\sigma}^{(n)}\big)\Bigg] M_{L_\tau-1,\sigma}\big(k_{0,\sigma}^{(P_{\sigma}n)}\leftarrow k_{L_\tau-1,\sigma}^{(n)}\big).
$$
采样时使用 $|w(X)|$ 作为概率权重，测量时将复相位/符号
$$
S(X)=\frac{w(X)}{|w(X)|}
$$
作为观测量（平均符号/相位因子 $\langle S\rangle_{|w|}$；实际可记录其实部与模长）。

permutation $P_{\sigma}$ 体现在虚时周期性边界条件上，
即 $k_{L_\tau,\sigma}^{(n)}=k_{0,\sigma}^{(P_{\sigma}n)}$。在具体实现中，我们进行局部的换位更新：任选两个粒子标号 $a\ne b$，提议以换位 $\tau_{ab}$ 更新 $P_{\sigma}'=\tau_{ab}\circ P_{\sigma}$（提议分布对称）。

由于 $P_{\sigma}$ 仅出现在边界片 $l=L_\tau-1$ 的两条矩阵元中，换位只影响这两条“边界链接”，因此接受率只涉及这两个因子：
$$
\mathcal{R}_{\text{perm}}
=\frac{|w(X')|}{|w(X)|}
=\frac{\big|M_{L_\tau-1,\sigma}(k_{0,\sigma}^{(P_{\sigma}b)}\leftarrow k_{L_\tau-1,\sigma}^{(a)})\big|\,\big|M_{L_\tau-1,\sigma}(k_{0,\sigma}^{(P_{\sigma}a)}\leftarrow k_{L_\tau-1,\sigma}^{(b)})\big|}
{\big|M_{L_\tau-1,\sigma}(k_{0,\sigma}^{(P_{\sigma}a)}\leftarrow k_{L_\tau-1,\sigma}^{(a)})\big|\,\big|M_{L_\tau-1,\sigma}(k_{0,\sigma}^{(P_{\sigma}b)}\leftarrow k_{L_\tau-1,\sigma}^{(b)})\big|}.
$$
接受-拒绝准则为 $A=\min(1,\mathcal{R}_{\text{perm}})$。测量时符号需额外乘以 $-1$，因为换位 $\tau_{ab}$ 改变了 $\text{sgn}(P_{\sigma})$。

更新 $K_{\sigma}$ 时，考虑最简单的单点局部更新：固定自旋 $\sigma$、粒子 $n$ 与虚时切片 $l$，提议
$$
k_{l,\sigma}^{(n)}\to k_{l,\sigma}^{\prime (n)}\equiv k_{l,\sigma}^{(n)}+q,\quad q\in\text{BZ}.
$$
为保持实现简单可令 $q$ 在布里渊区内均匀抽样（或从预先计算的 $|W_{l,\sigma}(q)|$ 分布抽样以提高接受率）。

该更新只影响与 $k_{l,\sigma}^{(n)}$ 相邻的两条“时间键”：
1) 片 $l$ 的矩阵元 $M_{l,\sigma}(k_{l+1,\sigma}^{(n)}\leftarrow k_{l,\sigma}^{(n)})$；
2) 片 $l-1$ 的矩阵元 $M_{l-1,\sigma}(k_{l,\sigma}^{(n)}\leftarrow k_{l-1,\sigma}^{(n)})$；
当 $l=0$ 或 $l=L_\tau-1$ 时，涉及到的边界键为 $M_{L_\tau-1,\sigma}(k_{0,\sigma}^{(P_{\sigma}n)}\leftarrow k_{L_\tau-1,\sigma}^{(n)})$。

因此接受率为（分情形写更直观）：
$$
\mathcal{R}_{k}
=\frac{|w(X')|}{|w(X)|}
=\begin{cases}
\dfrac{|M_{l,\sigma}(k_{l+1,\sigma}^{(n)}\!\leftarrow\! k_{l,\sigma}^{\prime (n)})|}{|M_{l,\sigma}(k_{l+1,\sigma}^{(n)}\!\leftarrow\! k_{l,\sigma}^{(n)})|}\,
\dfrac{|M_{l-1,\sigma}(k_{l,\sigma}^{\prime (n)}\!\leftarrow\! k_{l-1,\sigma}^{(n)})|}{|M_{l-1,\sigma}(k_{l,\sigma}^{(n)}\!\leftarrow\! k_{l-1,\sigma}^{(n)})|}, & 1\le l\le L_\tau-2;\\[8pt]
\dfrac{|M_{0,\sigma}(k_{1,\sigma}^{(n)}\!\leftarrow\! k_{0,\sigma}^{\prime (n)})|}{|M_{0,\sigma}(k_{1,\sigma}^{(n)}\!\leftarrow\! k_{0,\sigma}^{(n)})|}\,
\dfrac{|M_{L_\tau-1,\sigma}(k_{0,\sigma}^{\prime (n)}\!\leftarrow\! k_{L_\tau-1,\sigma}^{(m)})|}{|M_{L_\tau-1,\sigma}(k_{0,\sigma}^{(n)}\!\leftarrow\! k_{L_\tau-1,\sigma}^{(m)})|}, & l=0,\ m=P_{\sigma}^{-1}(n);\\[8pt]
\dfrac{|M_{L_\tau-1,\sigma}(k_{0,\sigma}^{(P_{\sigma}n)}\!\leftarrow\! k_{L_\tau-1,\sigma}^{\prime (n)})|}{|M_{L_\tau-1,\sigma}(k_{0,\sigma}^{(P_{\sigma}n)}\!\leftarrow\! k_{L_\tau-1,\sigma}^{(n)})|}\,
\dfrac{|M_{L_\tau-2,\sigma}(k_{L_\tau-1,\sigma}^{\prime (n)}\!\leftarrow\! k_{L_\tau-2,\sigma}^{(n)})|}{|M_{L_\tau-2,\sigma}(k_{L_\tau-1,\sigma}^{(n)}\!\leftarrow\! k_{L_\tau-2,\sigma}^{(n)})|}, & l=L_\tau-1.
\end{cases}
$$
同样 $A=\min(1,\mathcal{R}_{k})$。时间索引对 $L_\tau$ 取模。

由于 $M_{l,\sigma}(k'\leftarrow k)$ 的模长可写为
$$
\big|M_{l,\sigma}(k'\leftarrow k)\big|=e^{-\frac{\Delta\tau}{2}(\varepsilon_{k'}+\varepsilon_k)}\frac{|W_{l,\sigma}(k-k')|}{V},
$$
故上式比值的 $V$ 因子会相互抵消；接受率只依赖能量因子与 $|W|$ 的比值。

Pauli 不相容原理：对每个切片 $l$ 和自旋 $\sigma$，单粒子动量占据数满足 $n_\sigma(k,l)\in\{0,1\}$。在实现中可通过“碰撞拒绝”保证：
- 初始化 $K_\sigma$ 时保证同一切片内 $k_{l,\sigma}^{(n)}$ 互不相同；
- 进行 $k$ 的局部提议后，若新 $k_{l,\sigma}^{\prime (n)}$ 已被其它 $m\ne n$ 占据，则直接拒绝该提议（无需计算矩阵元比值）。

符号/相位测量（细化）：
- 将采样权重定义为 $p(X)=|w(X)|$，Metropolis 接受率使用 $p$ 的比值，不含任何相位或符号。
- 观测量定义为
  $S(X)=\text{sgn}(P_{\uparrow})\text{sgn}(P_{\downarrow})\exp[i\,\Phi(X)]$，其中
  $$\Phi(X)=\sum_{\sigma}\sum_{n}\Bigg[\sum_{l=0}^{L_\tau-2}\arg M_{l,\sigma}\big(k_{l+1,\sigma}^{(n)}\leftarrow k_{l,\sigma}^{(n)}\big)+\arg M_{L_\tau-1,\sigma}\big(k_{0,\sigma}^{(P_{\sigma}n)}\leftarrow k_{L_\tau-1,\sigma}^{(n)}\big)\Bigg].$$
- 实现时无需每步全量重算 $\Phi$，可增量维护：
  - $k$ 局部更新：仅更新涉及的两条（或边界时一条+一条边界）链接的相位差；将 $S\gets S\cdot e^{i\Delta\Phi_k}$。
  - permutation 换位：仅更新两条边界链接的相位差；并将 $S\gets (-1)\cdot S\cdot e^{i\Delta\Phi_{\text{perm}}}$。
- 统计量：在以 $p(X)$ 为权重的 Markov 链上，估计平均符号/相位
  $\langle S\rangle_p=\frac{1}{N}\sum_j S(X_j)$，可分别输出 $\operatorname{Re}\langle S\rangle,\operatorname{Im}\langle S\rangle$ 与 $|\langle S\rangle|$；误差按标准自相关校正的方差评估。

相位增量举例：
- $k$ 更新（内片 $1\le l\le L_\tau-2$）：
  $$\Delta\Phi_k=\arg\frac{M_{l,\sigma}(k_{l+1}^{(n)}\leftarrow k'_{l}{}^{(n)})}{M_{l,\sigma}(k_{l+1}^{(n)}\leftarrow k_{l}^{(n)})}
  +\arg\frac{M_{l-1,\sigma}(k'_{l}{}^{(n)}\leftarrow k_{l-1}^{(n)})}{M_{l-1,\sigma}(k_{l}^{(n)}\leftarrow k_{l-1}^{(n)})}.$$
- $k$ 更新（边界 $l=0$）：
  若 $m=P_{\sigma}^{-1}(n)$，则
  $$\Delta\Phi_k=\arg\frac{M_{0,\sigma}(k_{1}^{(n)}\leftarrow k'_{0}{}^{(n)})}{M_{0,\sigma}(k_{1}^{(n)}\leftarrow k_{0}^{(n)})}
  +\arg\frac{M_{L_\tau-1,\sigma}(k'_{0}{}^{(n)}\leftarrow k_{L_\tau-1}^{(m)})}{M_{L_\tau-1,\sigma}(k_{0}^{(n)}\leftarrow k_{L_\tau-1}^{(m)})}.$$
- $k$ 更新（边界 $l=L_\tau-1$）：
  $$\Delta\Phi_k=\arg\frac{M_{L_\tau-1,\sigma}(k_{0}^{(P n)}\leftarrow k'_{L_\tau-1}{}^{(n)})}{M_{L_\tau-1,\sigma}(k_{0}^{(P n)}\leftarrow k_{L_\tau-1}^{(n)})}
  +\arg\frac{M_{L_\tau-2,\sigma}(k'_{L_\tau-1}{}^{(n)}\leftarrow k_{L_\tau-2}^{(n)})}{M_{L_\tau-2,\sigma}(k_{L_\tau-1}^{(n)}\leftarrow k_{L_\tau-2}^{(n)})}.$$
- permutation 换位 $\tau_{ab}$：
  $$\Delta\Phi_{\text{perm}}=\arg\frac{M_{L_\tau-1,\sigma}(k_{0}^{(P b)}\leftarrow k_{L_\tau-1}^{(a)})}{M_{L_\tau-1,\sigma}(k_{0}^{(P a)}\leftarrow k_{L_\tau-1}^{(a)})}
  +\arg\frac{M_{L_\tau-1,\sigma}(k_{0}^{(P a)}\leftarrow k_{L_\tau-1}^{(b)})}{M_{L_\tau-1,\sigma}(k_{0}^{(P b)}\leftarrow k_{L_\tau-1}^{(b)})}.$$


### 辅助场 Gibbs 更新（半步波函数与磁化）

- 定义半步传播因子 $D(k)=\exp\left[-\frac{\Delta\tau}{2}\varepsilon_k\right]$，并以动量占据 $n_{\sigma,l}(k)$ 构造半步波函数
  $$
  \psi_{\sigma,l}(r)
  =\frac{1}{\sqrt{V}}\sum_k D(k)\,n_{\sigma,l}(k)\,e^{i k\cdot r},\qquad
  \psi_{\sigma,l+1}(r)
  =\frac{1}{\sqrt{V}}\sum_k D(k)\,n_{\sigma,l+1}(k)\,e^{i k\cdot r},
  $$
  实现中通过 IFFT（单位归一化）快速得到 $\psi$。
- 邻近时间片的重叠给出实空间粒子数
  $$
  n_{il\sigma} = \operatorname{Re}\!\left[\psi_{\sigma,l+1}(r_i)\,\psi_{\sigma,l}^\*(r_i)\right],\qquad
  m_{il}=n_{il\uparrow}-n_{il\downarrow}.
  $$
- 条件分布为独立的 Bernoulli（热浴）：
  $$
  P(s_{il}=+1)=\frac{1+\tanh(\lambda m_{il})}{2},\qquad
  P(s_{il}=-1)=1-P(s_{il}=+1).
  $$
- 一次 Gibbs sweep 遍历全部 $(i,l)$，按上述概率重采样 $s_{il}$。若某时间片发生翻转，则需：
  1. 重新计算该片的 $W_{l,\sigma}(q)$ 并更新任何依赖 $|W|$ 的提议表；
  2. 对所有穿过该时间片的链路，重新评估 $M_{l,\sigma}$ 的模与相位，累加
     $\Delta\log|w|$ 与 $\Delta\Phi$，从而保持全局权重与测量的连贯性。
- 在当前实现中，每个 Monte Carlo sweep 末执行一次完整的辅助场 Gibbs 更新；未来若需要，可再将其穿插于动量或 permutation 更新之间。

实现要点与建议：
- 辅助场 Gibbs 更新：模拟开始仍会根据配置生成初始 $s_{il}$ 并预存 $W_{l,\sigma}(q)$；随后每个 sweep 末按热浴分布重采样所有 $s_{il}$，同步刷新 $W$、动量提议分布以及对应的 $\log|w|/\Phi$ 增量。若在单点 sweep 内需要回滚（例如遇到零幅度链接），保持原场即可继续后续时间片。
- 初始化：可取 $P_{\sigma}=\text{id}$，并将 $k_{l,\sigma}^{(n)}$ 在 BZ 上均匀初始化，且对每个切片 $l$、自旋 $\sigma$ 确保不同粒子 $n$ 的动量互不相同（满足 Pauli 约束）。
- 更新日程：一次 sweep 可包含若干次 $K$ 局部更新（遍历 $l,n,\sigma$ 或随机抽样若干对 $(l,n,\sigma)$），以及若干次 permutation 的二体换位尝试；必要时可加入“循环移动”或“洗牌”以改善遍历性。
- 采样权重：Metropolis 接受率一律使用 $|w|$ 的比值；$\text{sgn}(P_{\uparrow})\text{sgn}(P_{\downarrow})$ 与 $\arg W$ 只进入观测量 $S(X)$。
- 观测量：记录 $S(X)=w/|w|=\text{sgn}(P_{\uparrow})\text{sgn}(P_{\downarrow})\exp[i\sum\arg W]$，并计算其样本平均与误差（可分开记录实部、虚部和模长）。

备注：若选择按 $|W_{l,\sigma}(q)|$ 的分布来提议 $q$，配合对称的“反向提议”权重，接受率中的 $|W|$ 因子可部分抵消，从而显著提高接受率；不过为保持程序最简，均匀提议亦可先行验证思路。

实现细节（已在代码中采用）：
- 对每个时间片与自旋，预先将 $|W_{l,\sigma}(q)|$ 归一化得到 $P_{l,\sigma}(q)=|W_{l,\sigma}(q)|/\sum_{q'}|W_{l,\sigma}(q')|$。若某片的 $|W|$ 全部为零，则退回到均匀分布避免失效。
- 在更新 $k_{l,\sigma}^{(n)}$ 时，先按 $P_{l,\sigma}$ 抽样 $q_{\text{new}}$，再令 $k_{l,\sigma}^{\prime (n)} = k_{l+1,\sigma}^{(n)} + q_{\text{new}}$（各分量模 $L$）。反向提议的概率为 $P_{l,\sigma}(q_{\text{old}})$，其中 $q_{\text{old}} = k_{l,\sigma}^{(n)} - k_{l+1,\sigma}^{(n)}$。
- Metropolis 指数使用
  \[
  \log \mathcal{R}_k = \log\frac{|w(X')|}{|w(X)|} + \log P_{l,\sigma}(q_{\text{old}}) - \log P_{l,\sigma}(q_{\text{new}})
  \]
  从而消去了前向链路上 $\frac{|W_{l,\sigma}(q_{\text{new}})|}{|W_{l,\sigma}(q_{\text{old}})|}$ 的部分。日志式接受判断只使用 $\log \mathcal{R}_k$，而用于测量的 $\log|w|$ 累计量仍仅记录 $\log\frac{|w(X')|}{|w(X)|}$ 的物理部分。
- 若配置项 `momentum_proposal` 设为 `"uniform"`，代码会跳过上述权重，退回到均匀提议以便做对照测试。
 
