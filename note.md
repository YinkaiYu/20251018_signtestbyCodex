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

配分函数写为：
$$
Z=\text{tr}\left( e^{-\beta H} \right)=
\frac{1}{N_{\uparrow}!}\frac{1}{N_{\downarrow}!}
\sum_{K_{\uparrow},P_{\uparrow}}\sum_{K_{\downarrow},P_{\downarrow}}
\text{sgn}(P_{\uparrow})\text{sgn}(P_{\downarrow})
\prod_{\sigma=\uparrow,\downarrow}
\prod_{n=1}^{N_{\sigma}}
\prod_{l=0}^{L_\tau-1}
\bra{k_{l+1,\sigma}^{(P_{\sigma}n)}}T_{l,\sigma}\ket{k_{l,\sigma}^{(n)}}
$$
其中
$$
\bra{k'}T_{l,\sigma}\ket{k}=e^{-\frac{\Delta\tau}{2}(\varepsilon_{k'}+\varepsilon_k)}\frac{W_{l,\sigma}(k-k')}{V}
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

蒙卡采样权重为
$$
w\left(K_{\uparrow},P_{\uparrow},K_{\downarrow},P_{\downarrow}\right)\propto
\text{sgn}(P_{\uparrow})\text{sgn}(P_{\downarrow})
\prod_{\sigma=\uparrow,\downarrow}
\prod_{n=1}^{N_{\sigma}}
\prod_{l=0}^{L_\tau-1}
\bra{k_{l+1,\sigma}^{(P_{\sigma}n)}}T_{l,\sigma}\ket{k_{l,\sigma}^{(n)}}
$$

蒙卡更新分别对 $K_{\sigma}$ 和 $P_{\sigma}$ 进行。注意：辅助场 $\{s_{il}\}$ 在模拟开始时一次性生成并固定，整个采样过程中不再更新（不采样辅助场），且全程不使用行列式；我们仅对动量空间世界线与 permutation 采样，观测量是本方法的平均符号/相位因子。

为便于记号，定义单片上的转移矩阵元
$$
M_{l,\sigma}(k'\leftarrow k)
=\bra{k'}T_{l,\sigma}\ket{k}
=e^{-\frac{\Delta\tau}{2}(\varepsilon_{k'}+\varepsilon_k)}\frac{W_{l,\sigma}(k-k')}{V}.
$$
对任意配置 $X\equiv (K_{\uparrow},P_{\uparrow};K_{\downarrow},P_{\downarrow})$，
其（可能为复数的）权重为
$$
w(X)=\text{sgn}(P_{\uparrow})\,\text{sgn}(P_{\downarrow})\,\prod_{\sigma}\prod_{n=1}^{N_\sigma}\prod_{l=0}^{L_\tau-1} M_{l,\sigma}\big(k_{l+1,\sigma}^{(P_{\sigma}n)}\leftarrow k_{l,\sigma}^{(n)}\big).
$$
采样时使用 $|w(X)|$ 作为概率权重，测量时将复相位/符号
$$
S(X)=\frac{w(X)}{|w(X)|}
$$
作为观测量（平均符号/相位因子 $\langle S\rangle_{|w|}$；实际可记录其实部与模长）。

permutation $P_{\sigma}$ 体现在虚时周期性边界条件上，
即 $k_{L_\tau,\sigma}^{(P_{\sigma}n)}=k_{0,\sigma}^{(n)}$。在具体实现中，我们进行局部的换位更新：任选两个粒子标号 $a\ne b$，提议以换位 $\tau_{ab}$ 更新 $P_{\sigma}'=\tau_{ab}\circ P_{\sigma}$（提议分布对称）。

此时仅与索引 $a,b$ 相关的因子发生变化，故接受率只需考虑它们沿虚时所有切片的贡献：
$$
\mathcal{R}_{\text{perm}}
=\frac{|w(X')|}{|w(X)|}
=\prod_{l=0}^{L_\tau-1}
\frac{\big|M_{l,\sigma}(k_{l+1,\sigma}^{(P_{\sigma}b)}\leftarrow k_{l,\sigma}^{(a)})\big|\,\big|M_{l,\sigma}(k_{l+1,\sigma}^{(P_{\sigma}a)}\leftarrow k_{l,\sigma}^{(b)})\big|}
{\big|M_{l,\sigma}(k_{l+1,\sigma}^{(P_{\sigma}a)}\leftarrow k_{l,\sigma}^{(a)})\big|\,\big|M_{l,\sigma}(k_{l+1,\sigma}^{(P_{\sigma}b)}\leftarrow k_{l,\sigma}^{(b)})\big|}.
$$
接受-拒绝准则为 $A=\min(1,\mathcal{R}_{\text{perm}})$。注意测量时的符号需额外乘以 $-1$，因为换位 $\tau_{ab}$ 会改变 $\text{sgn}(P_{\sigma})$ 的符号。

更新 $K_{\sigma}$ 时，考虑最简单的单点局部更新：固定自旋 $\sigma$、粒子 $n$ 与虚时切片 $l$，提议
$$
k_{l,\sigma}^{(n)}\to k_{l,\sigma}^{\prime (n)}\equiv k_{l,\sigma}^{(n)}+q,\quad q\in\text{BZ}.
$$
为保持实现简单可令 $q$ 在布里渊区内均匀抽样（或从预先计算的 $|W_{l,\sigma}(q)|$ 分布抽样以提高接受率）。

该更新只影响两条相邻“时间键”：
1) 片 $l$ 的矩阵元 $M_{l,\sigma}(k_{l+1,\sigma}^{(P_{\sigma}n)}\leftarrow k_{l,\sigma}^{(n)})$（$k$ 作为出射动量出现在 ket 上）；
2) 片 $l-1$ 的矩阵元 $M_{l-1,\sigma}(k_{l,\sigma}^{(P_{\sigma}i)}\leftarrow k_{l-1,\sigma}^{(i)})$，其中 $i=P_{\sigma}^{-1}(n)$（$k$ 作为入射动量出现在 bra 上）。

因此接受率为
$$
\mathcal{R}_{k}
=\frac{|w(X')|}{|w(X)|}
=\frac{\big|M_{l,\sigma}(k_{l+1,\sigma}^{(P_{\sigma}n)}\leftarrow k_{l,\sigma}^{\prime (n)})\big|}{\big|M_{l,\sigma}(k_{l+1,\sigma}^{(P_{\sigma}n)}\leftarrow k_{l,\sigma}^{(n)})\big|}
\times
\frac{\big|M_{l-1,\sigma}(k_{l,\sigma}^{\prime (n)}\leftarrow k_{l-1,\sigma}^{(i)})\big|}{\big|M_{l-1,\sigma}(k_{l,\sigma}^{(n)}\leftarrow k_{l-1,\sigma}^{(i)})\big|},\quad i=P_{\sigma}^{-1}(n),
$$
同样 $A=\min(1,\mathcal{R}_{k})$。当 $l=0$ 或 $l=L_\tau-1$ 时，时间索引按模 $L_\tau$ 处理。

由于 $M_{l,\sigma}(k'\leftarrow k)$ 的模长可写为
$$
\big|M_{l,\sigma}(k'\leftarrow k)\big|=e^{-\frac{\Delta\tau}{2}(\varepsilon_{k'}+\varepsilon_k)}\frac{|W_{l,\sigma}(k-k')|}{V},
$$
故上式比值的 $V$ 因子会相互抵消；接受率只依赖能量因子与 $|W|$ 的比值。测量时的相位增量则由 $\arg W_{l,\sigma}(\cdot)$ 的变化累积：一次 $k$ 局部更新会令
$$
\Delta\Theta=\arg W_{l,\sigma}\big(k_{l,\sigma}^{(n)}-k_{l+1,\sigma}^{(P_{\sigma}n)}\big)\Big|_{k\to k'}
+\arg W_{l-1,\sigma}\big(k_{l-1,\sigma}^{(i)}-k_{l,\sigma}^{(n)}\big)\Big|_{k\to k'}
-\text{(旧相位)},\quad i=P_{\sigma}^{-1}(n),
$$
并在 $S(X)$ 中以 $e^{i\Delta\Theta}$ 更新复相位因子。

实现要点与建议：
- 固定辅助场：在模拟开始生成 $s_{il}=\pm1$（独立同分布即可），并据此通过 FFT 预计算所有片的 $W_{l,\sigma}(q)$、其模 $|W|$ 与相位 $\arg W$；模拟过程中不再更新 $s_{il}$。
- 初始化：可取 $P_{\sigma}=\text{id}$，并将 $k_{l,\sigma}^{(n)}$ 在 BZ 上独立均匀初始化（越简单越好，权重非零即可）。
- 更新日程：一次 sweep 可包含若干次 $K$ 局部更新（遍历 $l,n,\sigma$ 或随机抽样若干对 $(l,n,\sigma)$），以及若干次 permutation 的二体换位尝试；必要时可加入“循环移动”或“洗牌”以改善遍历性。
- 采样权重：Metropolis 接受率一律使用 $|w|$ 的比值；$\text{sgn}(P_{\uparrow})\text{sgn}(P_{\downarrow})$ 与 $\arg W$ 只进入观测量 $S(X)$。
- 观测量：记录 $S(X)=w/|w|=\text{sgn}(P_{\uparrow})\text{sgn}(P_{\downarrow})\exp[i\sum\arg W]$，并计算其样本平均与误差（可分开记录实部、虚部和模长）。

备注：若选择按 $|W_{l,\sigma}(q)|$ 的分布来提议 $q$，配合对称的“反向提议”权重，接受率中的 $|W|$ 因子可部分抵消，从而显著提高接受率；不过为保持程序最简，均匀提议亦可先行验证思路。
