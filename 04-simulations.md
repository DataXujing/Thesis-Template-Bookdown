
# 数值模拟 {#simulations}

空间广义线性混合效应模型在广义线性混合效应模型基础上添加了与空间位置相关的随机效应，这种随机效应在文献中常称为空间效应 [@Diggle1998]。 它与采样点的位置、数量都有关系， 其中采样点的位置决定空间过程的协方差结构， 而采样点的数量决定空间效应的维度，从而导致空间广义线性混合效应模型比普通的广义线性混合效应模型复杂。作为过渡，我们在第 \@ref(sim-one-gp) 和 \@ref(sim-two-gp) 节模拟了一维和二维平稳高斯过程。 第 \@ref(sim-sglmm) 节模拟 SGLMM 模型， 分两个小节展开叙述，第 \@ref(sim-binomal-sglmm) 小节模拟响应变量服从二项分布的情形，  第 \@ref(possion-sglmm) 小节模拟响应变量服从泊松分布的情形，在这两个小节里， 比较了蒙特卡罗极大似然算法，低秩近似算法，贝叶斯马尔科夫链蒙特卡罗算法和贝叶斯 STAN-MCMC 算法的性能差异。

模拟空间广义线性混合效应模型 \@ref(eq:sim-sglmm) 分三节进行，第一节模拟空间高斯过程 $S(x)$，第二节模拟响应变量 $Y$ 服从二项分布 ，第三节模拟响应变量 $Y$ 服从正态分布。空间高斯过程在模型 \@ref(eq:sim-sglmm) 中作为随机效应存在，不同于一般随机效应的地方在于它与样本的空间位置有关，一般地，假定 $S(x)$ 服从 $N$ 元高斯分布 $N(\mu_{S},G)$，$x = (d_1,d_2) \in \mathbb{R}^2$， $\mu_{S} = \mathbf{0}_{N\times1}$， $G_{(ij)} = \mathrm{Cov}(S(x_i),S(x_j))=\sigma^2*\rho(u_{ij})$， $S(x)$ 的相关函数 $\rho(u_{ij}) = \exp(-u_{ij}/\phi), u_{ij} \equiv \|x_{i}-x_{j}\|_2$，其中 $\phi$ 和 $\sigma^2$ 都是未知待估参数。可见采样点的位置 $(d_1,d_2)$ 和相关函数 $\rho(u)$ 一起决定空间高斯过程的形式，并且 $S(x)$ 的维度就是采样点的数目 $N$。这样通常导致空间效应的维度比协变量的数目大很多，模型 \@ref(eq:sim-sglmm) 的估计问题也比一般的广义线性混合效应模型困难。

\begin{equation}
g(\mu_i) = d(x_i)'\beta + S(x_i) + Z_i (\#eq:sim-sglmm)
\end{equation}

## 平稳空间高斯过程 {#spatial-gaussian-processes}

### 一维平稳空间高斯过程 {#sim-one-gp}

一维情形下，平稳高斯过程 $S(x)$ 的协方差函数采用幂指数型，见公式 \@ref(eq:cov-exp-quad)，当 $\kappa =1$ 时，为指数型，见公式 \@ref(eq:cov-exp-quad)。分 $\kappa =1$ 和 $\kappa =1$，模拟两个一维平稳空间高斯过程，协方差参数均为 $\sigma^2 = 1$，$\phi = 0.15$，均值向量都是 $\mathbf{0}$，在 $[-2,2]$ 的区间上，产生 2000 个服从均匀分布的随机数，由这些随机数的未知和协方差函数公式 \@ref(eq:cov-exp) 或 \@ref(eq:cov-exp-quad) 计算得到 2000 维的高斯分布的协方差矩阵 $G$，为保证协方差矩阵的正定性，在矩阵对角线上添加扰动 $1 \times 10^{-12}$，然后即可根据 Cholesky 分解该对称正定矩阵，得到下三角块 $L$，使得 $G = L \times L^{\top}$，再产生 2000 个服从标准正态分布的随机向量 $\eta$，而 $L\eta$ 即为所需的服从平稳高斯过程的一组随机数。

\begin{align}
\mathsf{Cov}(S(x_i), S(x_j)) & = \sigma^2 \exp\big\{ - \frac{|x_{i} - x_{j}|}{ \phi } \big\}  (\#eq:cov-exp) \\
\mathsf{Cov}(S(x_i), S(x_j)) & = \sigma^2 \exp\big\{ -\big( \frac{ |x_{i} - x_{j}| }{ \phi } \big) ^ {\kappa} \big\}, 0 < \kappa \leq 2  (\#eq:cov-exp-quad) 
\end{align}


\begin{figure}

{\centering \subfloat[平稳空间高斯过程 $S(x)$ 的协方差函数是指数型，均值向量为 $\mathbf{0}$，协方差参数 $\sigma^2 = 1$，$\phi = 0.15$(\#fig:one-dim-gp1)]{\includegraphics[width=0.7\linewidth]{figures/one-dim-gp-exp} }\\\subfloat[平稳空间高斯过程 $S(x)$ 的协方差函数是幂二次指数型，均值向量为 $\mathbf{0}$，协方差参数 $\sigma^2 = 1$，$\phi = 0.15$，$\kappa=2$(\#fig:one-dim-gp2)]{\includegraphics[width=0.7\linewidth]{figures/one-dim-gp-exp-quad} }

}

\caption{(ref:one-dim-gp)}(\#fig:one-dim-gp)
\end{figure}

(ref:one-dim-gp) 模拟一维平稳空间高斯过程，协方差函数分别为指数型 \@ref(eq:cov-exp) 和幂二次指数型 \@ref(eq:cov-exp-quad)，均值为 $\mathbf{0}$，协方差参数 $\sigma^2 = 1$，$\phi = 0.15$，横坐标表示采样的位置，纵坐标是目标值 $S(x)$，图中 2000 个灰色点表示服从相应随机过程的随机数，橘黄色点是从中随机选择的 36 个点。

根据定理 \@ref(thm:stationary-mean-square-properties)，指数型协方差函数的平稳高斯过程在原点连续但是不可微，而幂二次指数型协方差函数在原点无穷可微，可微性越好图像上表现越光滑，对比图 \@ref(fig:one-dim-gp) 的两个子图， 可以看出，在协方差参数 $\sigma^2 = 1$，$\phi = 0.15$ 相同的情况下，$\kappa$ 越大越光滑。

### 二维平稳空间高斯过程 {#sim-two-gp}

二维情形下，在规则平面上模拟平稳高斯过程 $\mathcal{S} = S(x), x \in \mathbb{R}^2$， 其均值向量为零向量 $\mathbf{0}$， 协方差函数为指数型 (见公式 \@ref(eq:cov-exp)) ，协方差参数 $\phi = 1, \sigma^2 = 1$。在单位平面区域为 $[0,1] \times [0,1]$ 模拟服从上述二维空间平稳高斯过程，不妨将此区域划分为 $6 \times 6$ 的小网格，而每个格点作为采样的位置，共计 36个采样点，在这些采样点上的观察值即为目标值 $S(x)$。 

类似本章第 \@ref(sim-one-gp) 节模拟一维平稳空间过程的步骤， 首先根据采样点位置坐标和协方差函数 \@ref(eq:cov-exp-quad) 计算得目标空间过程的 $\mathcal{S}$ 协方差矩阵 $G$，然后使用 R 包 **MASS** 提供的 `mvrnorm` 函数产生多元正态分布随机数，与 \@ref(sim-one-gp) 节不同的是这里采用特征值分解，即 $G = L\Lambda L^{\top}$，与 Cholesky 分解相比，特征值分解更稳定些，但是 Cholesky 分解更快，Stan 即采用此法，后续过程与一维模拟一致。模拟获得的随机数用图 \@ref(fig:sim-two-gp) 表示， 格点上的值即为平稳空间高斯过程在该点的取值 (为方便显示，已四舍五入保留两位小数)。

\begin{figure}

{\centering \subfloat[在单位区域的网格点上采样(\#fig:sim-two-gp1)]{\includegraphics[width=0.45\linewidth]{04-simulations_files/figure-latex/sim-two-gp-1} }\subfloat[在单位区域上随机采样(\#fig:sim-two-gp2)]{\includegraphics[width=0.45\linewidth]{04-simulations_files/figure-latex/sim-two-gp-2} }

}

\caption{模拟二维平稳空间高斯过程，自相关函数为指数形式，水平方向为横坐标，垂直方向为纵坐标，图中的橘黄色点是采样的位置，其上的数字是目标值 $S(x)$}(\#fig:sim-two-gp)
\end{figure}


同 \@ref(sim-one-gp) 节，二维平稳空间高斯过程 $S(x)$ 的协方差函数也可以为更一般的梅隆型，如公式 \@ref(eq:exp-matern) 所示。

\begin{equation}
\rho(u) = \sigma^2 \{ 2^{\kappa -1} \Gamma(\kappa) \}^{-1}( u/\phi )^{\kappa} \mathcal{K}_{\kappa}( u / \phi ) (\#eq:exp-matern)
\end{equation}

\noindent 且在区域 $[0,1] \times [0,1]$ 上也可以随机采点，如图 \@ref(fig:sim-two-gp) 的右子图所示。

模拟平稳空间高斯过程的其它 R 包实现方法： Paulo J. Ribeiro Jr. 和 Peter J. Diggle 开发了 **geoR** 包 [@geoR2001]，提供的 `grf` 函数除了实现 Cholesky 分解，还实现了奇异值分解，特征值分解协方差矩阵的算法。当采样点不太多时，Cholesky 分解已经足够好，下面的第 \@ref(sim-sglmm) 节对平稳空间高斯过程的数值模拟即采用此法，当采样点很多，为了加快模拟的速度，可以选用 Martin Schlather 等开发的 **RandomFields** 包 [@RandomFields2015]，内置的 `GaussRF` 函数实现了用高斯马尔科夫随机场近似平稳空间高斯过程的算法，此外，Håvard Rue 等 (2009年) [@Rue2009] 也实现了从平稳高斯过程到高斯马尔科夫随机场的近似算法，开发了更为高效的 INLA 程序库 [@INLA2015]，其内置的近似程序得到了广泛的应用 [@Virgilio2018;@Blangiardo2015;@Faraway2018]。


## 空间广义线性混合效应模型 {#sim-sglmm}

### 响应变量服从二项分布 {#sim-binomal-sglmm}

响应变量服从二项分布 $Y_{i} \sim \mathrm{Binomal}(m_{i},p(x_{i}))$，在位置 $x_i$ 处，以概率 $p(x_i)$ 重复抽取了 $m_i$ 个样本，总样本数 $M=\sum_{i=1}^{N}m_i$，$N$ 是采样点的个数，二项空间广义线性混合效应模型为 \@ref(eq:binom-SGLMM)，联系函数为 $g(\mu_i) = \log\{\frac{p(x_i)}{1-p(x_i)}\}$，$S(x)$ 是均值为 $\mathbf{0}$，协方差函数为 $\mathrm{Cov}(S(x_i),S(x_j)) = \sigma^2 \big\{2^{\kappa-1}\Gamma(\kappa)\big\}^{-1}(u/\phi)^{\kappa}K_{\kappa}(u/\phi), \kappa = 0.5$ 的平稳空间高斯过程。

\begin{equation}
g(\mu_i) = \log\{\frac{p(x_i)}{1-p(x_i)}\} = \alpha + S(x_i) (\#eq:binom-SGLMM)
\end{equation}

固定效应参数 $\alpha = 0$，协方差参数记为 $\boldsymbol{\theta} = (\sigma^2, \phi) = (0.5, 0.2)$，采样点数目为 $N = 64$，每个采样点抽取的样本数 $m_i = 4, i = 1, 2, \ldots, 64$。首先模拟平稳空间高斯过程 $S(x)$，在单位区域 $[0,1] \times [0,1]$ 划分为 $8 \times 8$ 的网格，格点选为采样位置，用 **geoR** 包提供的 `grf` 函数产生协方差参数为 $\boldsymbol{\theta} = (\sigma^2,\phi) = (0.5, 0.2)$ 的平稳空间高斯过程，由公式 \@ref(eq:binom-SGLMM) 可知 $p(x_i) = \exp(S(x_i))/(1 + \exp(S(x_i)))$， 即每个格点处二项分布的概率值，然后依此概率，由 `rbinom` 函数产生服从二项分布的观察值 $Y_i$，$Y_i$ 的取值范围为 $0, 1, 2, 3, 4$，模拟的数据集可以用图 \@ref(fig:binom-without-nugget-geoRglm) 直观表示。

\begin{figure}

{\centering \includegraphics[width=0.9\linewidth]{figures/binom-without-nugget-geoRglm} 

}

\caption{左图表示二维规则平面上的平稳空间高斯过程，格点是采样点的位置，其上的数字是 $p(x)$ 的值，已经四舍五入保留两位小数，右图表示观察值 $Y$ 随空间位置的变化，格点上的值即为观察值 $Y$，图中的两个圈分别是第1个(左下)和第29个(右上)采样点}(\#fig:binom-without-nugget-geoRglm)
\end{figure}

基于 Langevin-Hastings 采样器实现的马尔科夫链蒙特卡罗算法，参数 $\alpha$ 的先验分布选均值为 0，方差为 1 的标准正态分布，参数 $\phi$ 的先验分布选期望为 0.2 的指数分布，参数 $\sigma^2$ 的先验分布是非中心的逆卡方分布，其非中心化参数为 0.5，自由度为 5，各参数的先验选择来自 Ole F. Christensen 和 Peter J. Ribeiro Jr. (2002年) [@geoRglm2002]。Langevin-Hastings 算法运行 110000 次迭代，前 10000 次迭代用作热身 (warm-up)，后 10 万次迭代里间隔 100 次迭代采样，获得关于参数 $\alpha,\phi,\sigma^2$ 的后验分布的样本，样本量是1000。

\begin{align}
\alpha   & \sim \mathcal{N}(0,1) \\
\phi     & \sim \mathrm{Exp}(0.2) \\
\sigma^2 & \sim \mathrm{Inv-}\chi^2(5,0.5)
\end{align}

Table: (\#tab:MCLH-vs-NUTS)  Langevin-Hastings 算法与 NUTS 算法的数值模拟比较，模型参数 $\alpha,\phi,\sigma^2$ 的估计值（后验分布的均值）、方差（后验分布的方差）和 5 个分位点（后验分布的 5 个分位点），采样点数目分别考虑了 $N = 36, 64, 81, 100$ 的情况，括号内为相应参数的真值

|        |  mean|   var|   2.5\%|    25\%|   50\%|   75\%| 97.5\%|   N  |
|:-------|-----:|-----:|-------:|-------:|------:|------:|------:|-----:|
|$\alpha$    | -0.354(0)  | 0.079| -0.938| -0.524| -0.361| -0.173| 0.215| 36 |
|$\phi$      |  0.121(0.2)| 0.006|  0.005|  0.055|  0.110|  0.180| 0.285| 36 |
|$\sigma^2$  |  0.683(0.5)| 0.147|  0.215|  0.408|  0.596|  0.850| 1.667| 36 |
|$\alpha$    | 0.003(0)  | 0.089| -0.596| -0.169| 0.013| 0.179| 0.609| 64 |
|$\phi$      | 0.194(0.2)| 0.004|  0.070|  0.145| 0.195| 0.250| 0.295| 64 |
|$\sigma^2$  | 0.656(0.5)| 0.096|  0.254|  0.449| 0.592| 0.781| 1.453| 64 |
|$\beta$    | -0.155(0)  | 0.044| -0.565| -0.284| -0.156| -0.03| 0.273| 81 |
|$\phi$     |  0.116(0.2)| 0.006|  0.005|  0.055|  0.105|  0.17| 0.280| 81 |
|$\sigma^2$ |  0.468(0.5)| 0.057|  0.180|  0.311|  0.414|  0.56| 1.129| 81 |

参数 $\alpha,\phi,\sigma^2$ 的贝叶斯估计没有显式表达式，通常用后验分布的均值作为参数的估计， 估计的精度或者说好坏用后验分布的方差衡量，而均方误差在参数估计取后验均值时，就是后验方差，所以表 \@ref(tab:MCLH-vs-NUTS) 不再提供估计的均方误差值，而是提供了 5 个后验分布的分位点，在 95\% 的置信水平下，样本分位点 0.025 和 0.975 的值组成了置信区间的上下界。64 个采样点处 $p(x_i), i = 1, \ldots, 64$ 的后验分布的均值、方差、标准差和 5个分位点见附表 \@ref(tab:LH-binom-SGLMM)


### 响应变量服从泊松分布 {#possion-sglmm}

响应变量 $Y$ 服从泊松分布，即 $Y_i \sim \mathrm{Possion}(\lambda(x_{i}))$，泊松空间广义线性混合效应模型 \@ref(eq:pois-SGLMM)

\begin{equation}
g(\mu_i) = \log(\lambda(x_i)) = \alpha + S(x_i) (\#eq:pois-SGLMM)
\end{equation}

\noindent 其中，$S(x)$ 是平稳空间高斯过程，其均值为 $\mathbf{0}$，协方差函数为 $\mathrm{Cov}(S(x_i),S(x_j)) = \sigma^2 \big\{2^{\kappa-1}\Gamma(\kappa)\big\}^{-1}(u/\phi)^{\kappa}K_{\kappa}(u/\phi)$，联系函数 $g(\mu_i) = \log\{\lambda(x_{i})\}$。

类似 \@ref(sim-binomal-sglmm) 小节产生随机数的过程，

Table: (\#tab:Pois-MCLV-vs-NUTS) 模型参数真值设置为 $\alpha = 0.5, \phi = 0.2, \sigma^2 = 2.0, \kappa = 1.5$，采样点数目分别为 $N=36,64,100$

|           |  true|   mean|    var|    2.5%|     25%|    50%|    75%|  97.5%|   N   |
|:----------|-----:|------:|------:|-------:|-------:|------:|------:|------:|------:|
|$\alpha$   |   0.5|  0.527|  0.418|  -0.759|  0.189|  0.514|  0.855|  1.864|  36|
|$\phi$     |   0.2|  0.401|  0.052|   0.100|  0.240|  0.360|  0.520|  0.960|  36|
|$\sigma^2$ |   2.0|  1.311|  0.660|   0.365|  0.766|  1.081|  1.584|  3.562|  36|
|$\alpha$   |   0.5|  0.866|  1.517|  -1.610|  0.059|  0.870|  1.666|  3.159|  64|
|$\phi$     |   0.2|  0.682|  0.073|   0.300|  0.480|  0.640|  0.820|  1.380|  64|
|$\sigma^2$ |   2.0|  3.932|  2.594|   1.667|  2.800|  3.642|  4.744|  7.740|  64|
|$\alpha$   |   0.5|  0.323|  0.657|  -1.449|  -0.124|  0.416|  0.812|  1.831|  100  |
|$\phi$     |   0.2|  0.617|  0.085|   0.220|   0.400|  0.560|  0.785|  1.320|  100  |
|$\sigma^2$ |   2.0|  1.479|  0.498|   0.545|   0.941|  1.352|  1.822|  3.195|  100  |


