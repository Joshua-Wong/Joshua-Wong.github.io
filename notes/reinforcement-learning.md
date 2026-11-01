## 强化学习的数学原理

[【强化学习的数学原理】课程：从零开始到透彻理解（完结）](https://www.bilibili.com/video/BV1sd4y167NS/?share_source=copy_web&vd_source=a0e8f83e10bb7f4fb25de145fdf7fa19)

前三章的主线是：先描述智能体如何与环境交互，再评价一个给定策略，最后刻画最优策略。对应的数学工具依次是马尔可夫决策过程、贝尔曼方程、贝尔曼最优方程。

**统一约定**：主要讨论有限状态、有限动作、时间齐次的无限时域折扣 MDP；奖励有界，折扣因子 $0\leq\gamma<1$。策略默认是平稳马尔可夫策略。后文关于唯一解、收敛和确定性最优策略的结论均在这些条件下使用。大写 $S_t,A_t,R_{t+1}$ 表示随机变量，小写 $s,a,r$ 表示具体取值；$V^\pi,Q^\pi$ 与教材中的 $v_\pi,q_\pi$ 含义相同。

### Chap 1: Basic Concepts

#### 1.1 智能体与环境（Agent and Environment）

强化学习研究智能体如何通过与环境交互，选择能够带来较高长期回报的行动。一次交互的时序为

$$
S_t \xrightarrow{\text{选择 }A_t} (R_{t+1},S_{t+1}).
$$

智能体在时刻 $t$ 观察状态 $S_t$，选择动作 $A_t$；环境随后给出奖励 $R_{t+1}$ 和下一状态 $S_{t+1}$。这里奖励下标采用 $t+1$，表示它是执行 $A_t$ 后得到的反馈，不是执行下一动作 $A_{t+1}$ 后的反馈。

需要区分两个问题：**环境规定采取行动后会发生什么，策略规定智能体选择什么行动。**

#### 1.2 状态与动作（State and Action）

- **State**：The status of the agent with respect to the environment. 状态用于描述智能体及环境中与决策有关的信息，例如位置、速度、剩余电量；若任务依赖时间，也可能需要包含当前时间或剩余步数。所有可能状态构成状态空间（State Space），记为 $\mathcal S$。
- **Action**：智能体能够采取的行动，例如向前或向后走。状态 $s$ 下允许采取的动作构成动作空间 $\mathcal A(s)$，它可以依赖当前状态。不能在某状态下执行的动作，不应混入该状态的求和或最大化运算。
- **State Transition**：智能体采取一个动作后，状态按照环境规律转移的过程。转移也可能返回原状态；“发生转移”不意味着状态本身一定改变。

状态不一定等于传感器的一次观测。例如，车辆的位置相同而速度不同，下一时刻的位置分布可能不同；仅用位置作为状态可能丢失必要信息。设计状态时，要尽量保留预测未来所需的信息，这与下文的马尔可夫性质有关。

#### 1.3 状态转移概率（State Transition Probability）

给定当前状态和动作，定义

$$
p(s'\mid s,a)=\Pr(S_{t+1}=s'\mid S_t=s,A_t=a).
$$

对每个合法状态动作对 $(s,a)$，有

$$
p(s'\mid s,a)\geq0,\qquad
\sum_{s'\in\mathcal S}p(s'\mid s,a)=1.
$$

- 确定性环境：某个下一状态的概率为 $1$，其余为 $0$。
- 随机环境：同一 $(s,a)$ 可能产生不同的下一状态。例如“向右走”可能因打滑而向上移动。

环境随机性与策略随机性是两回事：即使动作选择完全确定，环境仍可能随机；即使环境完全确定，智能体也可以随机选择动作。

#### 1.4 策略（Policy）

策略是智能体根据状态选择动作的规则。平稳随机策略写为

$$
\pi(a\mid s)=\Pr(A_t=a\mid S_t=s),\qquad
\sum_{a\in\mathcal A(s)}\pi(a\mid s)=1.
$$

“平稳”表示这条规则不显式依赖 $t$；“马尔可夫”表示选择动作只依赖当前状态，而不额外依赖完整历史。

确定性策略是随机策略的特例：对每个状态指定一个动作 $\mu(s)$，使

$$
\pi(a\mid s)=
\begin{cases}
1,&a=\mu(s),\\
0,&a\neq\mu(s).
\end{cases}
$$

策略是覆盖各个状态的决策规则，不是预先排好的一串动作。环境可能把智能体带到不同状态，策略必须说明到达这些状态后如何继续行动。

#### 1.5 奖励（Reward）

奖励是环境给出的数值反馈，表示当前一步的得失。它可以为正、为负或为零，也可以是随机变量。为避免把随机奖励与平均奖励混淆，记

$$
\bar r(s,a)=\mathbb E[R_{t+1}\mid S_t=s,A_t=a].
$$

若奖励取离散值，也可以写成 $\bar r(s,a)=\sum_r p(r\mid s,a)r$。

更完整的环境模型使用下一状态与奖励的联合分布：

$$
p(s',r\mid s,a)
=\Pr(S_{t+1}=s',R_{t+1}=r\mid S_t=s,A_t=a).
$$

它给出

$$
p(s'\mid s,a)=\sum_r p(s',r\mid s,a),\qquad
\bar r(s,a)=\sum_{s',r}p(s',r\mid s,a)r.
$$

这样允许奖励与下一状态相关，不必假设二者独立。连续奖励时，将关于奖励的求和换成相应积分即可。

**奖励描述目标，不直接指定正确动作。** 一步奖励较大，不代表长期更好；为获得将来的收益，有时需要先付出代价。奖励设计若偏离真正目标，即使求得该奖励下的最优策略，也可能得到不符合实际意图的行为。

#### 1.6 轨迹、回合与回报（Trajectory, Episode and Return）

一条轨迹是交互产生的序列：

$$
S_0,A_0,R_1,S_1,A_1,R_2,S_2,\ldots
$$

它是一次随机实现，不等同于环境模型或策略。若任务在终止时刻 $T$ 结束，从起点到终点的一次完整交互称为一个回合（Episode）；没有自然结束点的任务称为持续型任务（Continuing Task）。

从时刻 $t$ 开始的折扣回报定义为

$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}
=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
$$

- $\gamma=0$：只考虑下一步奖励。
- $\gamma$ 越接近 $1$：相同远期奖励的权重越大。
- 若 $|R_t|\leq R_{\max}$，则 $|G_t|\leq R_{\max}/(1-\gamma)$，无限和绝对收敛。

因此，折扣既表达对远期收益的权重，也为无限时域分析提供收敛保证。截断到前 $N$ 项时，尾项误差至多为 $R_{\max}\gamma^N/(1-\gamma)$，这正是几何级数余项估计。

对回合型任务，回报可写为

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1},\qquad G_T=0.
$$

也可以在终止后补上一个奖励恒为零的吸收状态，使有限和与无限和写法统一。进入终止状态那一步的奖励仍然计入回报。若取 $\gamma=1$，则需另外保证回报及其期望存在，不能直接套用后文的折扣收敛证明。

#### 1.7 马尔可夫决策过程（Markov Decision Process）

可将环境及目标记为

$$
\mathcal M=(\mathcal S,\{\mathcal A(s)\}_{s\in\mathcal S},p,\gamma),
$$

其中 $p$ 指联合转移奖励模型 $p(s',r\mid s,a)$，已经包含奖励信息；也常把转移模型与奖励函数分开列出。策略是作用在该 MDP 上的决策规则，一般不作为待固定的环境组成部分。

令历史为 $H_t=(S_0,A_0,R_1,\ldots,S_t)$。马尔可夫性质要求

$$
\Pr(S_{t+1}=s',R_{t+1}=r\mid H_t,A_t)
=p(s',r\mid S_t,A_t).
$$

含义是：**给定当前状态和动作后，更早的历史不再提供预测下一状态和奖励所需的额外信息。** 它不是说“未来与过去无关”，而是说当前状态已经概括了相关历史。

“时间齐次”则要求右边的模型不随 $t$ 改变。马尔可夫性与时间齐次性是不同假设。固定一个平稳策略后，状态转移概率变成

$$
P^\pi(s,s')=\sum_{a\in\mathcal A(s)}\pi(a\mid s)p(s'\mid s,a),
$$

于是受控的 MDP 诱导出一个马尔可夫链。第二章就用这个转移矩阵评价策略。

#### 1.8 贯穿算例：现在获利还是先升级

考虑一个两状态持续型任务，取 $\gamma=0.9$：

- 状态 $s_1$ 表示普通模式。有两个动作：$a_{\mathrm{stay}}$ 获得奖励 $1$ 并留在 $s_1$；$a_{\mathrm{up}}$ 获得奖励 $0$ 并转入 $s_2$。
- 状态 $s_2$ 表示高收益模式。只有一个动作 $a_{\mathrm{earn}}$，每步获得奖励 $2$ 并留在 $s_2$。
- 所有转移和奖励均确定；不设终止状态。

一直留在 $s_1$，回报为 $1/(1-0.9)=10$；立刻升级后持续获利，回报为 $0+0.9\times2/(1-0.9)=18$。升级的即时奖励较低，但长期回报更高。

下一章先不寻找最优策略，而是评价一个给定随机策略：在 $s_1$ 以各 $1/2$ 的概率选择两个动作，在 $s_2$ 选择唯一动作。

### Chap 2: Bellman Equation

#### 2.1 状态价值（State Value）

同一状态出发，因策略或环境的随机性，不同轨迹可能有不同回报。用期望概括一个给定策略的长期表现：

$$
V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s].
$$

这里的含义是“从 $s$ 开始，此后一直按 $\pi$ 行动，能够获得多少期望折扣回报”。下标 $\pi$ 提醒我们：期望同时包含动作选择和环境反馈的随机性。

三个容易混淆的量：

- $R_{t+1}$：一次交互中的即时奖励。
- $G_t$：一条轨迹上未来奖励的折扣和，是随机变量。
- $V^\pi(s)$：从状态 $s$ 出发的回报期望，是给定策略下的确定数值。

在时间齐次环境和固定平稳策略下，$V^\pi(s)$ 不必带时间下标；有限时域任务若剩余时间影响决策，则需写成 $V_t^\pi(s)$，或将剩余时间并入状态。

#### 2.2 从回报递推推导贝尔曼方程

首先把回报的第一项分离出来：

$$
\begin{aligned}
G_t
&=R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+\cdots)\\
&=R_{t+1}+\gamma G_{t+1}.
\end{aligned}
$$

对当前状态求条件期望，利用期望的线性性：

$$
V^\pi(s)=\mathbb E_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s].
$$

再按当前动作、下一状态和奖励展开全期望：

$$
V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma\mathbb E_\pi[G_{t+1}\mid S_t=s,A_t=a,S_{t+1}=s',R_{t+1}=r]\right].
$$

由于马尔可夫性质，且下一步起仍使用同一策略，括号内的未来回报期望等于 $V^\pi(s')$。因此

$$
V^\pi(s)=\sum_{a\in\mathcal A(s)}\pi(a\mid s)
\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma V^\pi(s')\right].
$$

将即时奖励取期望，得到更常用的形式：

$$
\boxed{
V^\pi(s)=\sum_{a\in\mathcal A(s)}\pi(a\mid s)
\left[\bar r(s,a)+\gamma\sum_{s'\in\mathcal S}p(s'\mid s,a)V^\pi(s')\right].
}
$$

这就是给定策略下的贝尔曼方程（Bellman Equation，也称 Bellman Expectation Equation）。两层平均分别对应“当前选哪个动作”和“环境把我带到哪里”。它把无限长的未来压缩为一步奖励和下一状态价值之间的关系。

**这里不需要奖励与下一状态独立。** 将联合形式拆成两项只用了期望的线性性；不能反过来把联合概率擅自写成两个边缘概率的乘积。

#### 2.3 矩阵形式与唯一解

设 $\mathcal S=\{s_1,\ldots,s_n\}$，将各状态价值排成列向量 $\mathbf v^\pi$。定义

$$
[\mathbf r^\pi]_i=\sum_a\pi(a\mid s_i)\bar r(s_i,a),\qquad
[P^\pi]_{ij}=\sum_a\pi(a\mid s_i)p(s_j\mid s_i,a).
$$

矩阵 $P^\pi$ 的行对应当前状态，列对应下一状态；各元素非负，每行和为 $1$。将 $n$ 个标量方程放在一起：

$$
\mathbf v^\pi=\mathbf r^\pi+\gamma P^\pi\mathbf v^\pi,
\qquad
(I-\gamma P^\pi)\mathbf v^\pi=\mathbf r^\pi.
$$

因此形式解为

$$
\boxed{\mathbf v^\pi=(I-\gamma P^\pi)^{-1}\mathbf r^\pi.}
$$

为什么可以求逆？利用矩阵无穷范数 $\|P^\pi\|_\infty=1$，有 $\|\gamma P^\pi\|_\infty=\gamma<1$，故 Neumann 级数收敛：

$$
(I-\gamma P^\pi)^{-1}=\sum_{k=0}^{\infty}\gamma^k(P^\pi)^k.
$$

这既证明解存在且唯一，又给出概率解释：

$$
\mathbf v^\pi=\mathbf r^\pi+\gamma P^\pi\mathbf r^\pi
+\gamma^2(P^\pi)^2\mathbf r^\pi+\cdots.
$$

其中 $(P^\pi)^k\mathbf r^\pi$ 的第 $i$ 个分量，是从 $s_i$ 出发、按策略经过 $k$ 次转移后，再获得的一步奖励的期望。

从数值分析角度，实际计算通常应求解线性方程组，而非显式形成逆矩阵。闭式表达式主要用于理论分析；状态很多或模型未知时，直接求解通常不可行。

#### 2.4 固定策略下的不动点迭代

定义贝尔曼期望算子

$$
T^\pi\mathbf v=\mathbf r^\pi+\gamma P^\pi\mathbf v.
$$

价值向量是它的不动点：$T^\pi\mathbf v^\pi=\mathbf v^\pi$。对任意两个向量，

$$
\|T^\pi\mathbf v-T^\pi\mathbf w\|_\infty
\leq\gamma\|\mathbf v-\mathbf w\|_\infty.
$$

所以反复代入 $\mathbf v_{k+1}=T^\pi\mathbf v_k$，从任意有限初值都收敛到 $\mathbf v^\pi$，并满足

$$
\|\mathbf v_k-\mathbf v^\pi\|_\infty
\leq\gamma^k\|\mathbf v_0-\mathbf v^\pi\|_\infty.
$$

这就是熟悉的压缩映射迭代。这里 $k$ 是求解器的迭代次数，不是环境中的时间 $t$。它评价的是固定策略，过程中并没有自动把策略变优。

#### 2.5 动作价值（Action Value）及其与状态价值的关系

定义

$$
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

准确理解是：从 $s$ 开始，**第一步指定执行 $a$，从下一步起再遵循 $\pi$**。即使 $\pi(a\mid s)=0$，也可以通过这种“指定第一步”的方式定义该动作价值。

由一步展开，

$$
\boxed{Q^\pi(s,a)=\bar r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V^\pi(s').}
$$

反过来，在状态 $s$ 按策略随机选择第一步动作：

$$
\boxed{V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a).}
$$

合并两式即可得到动作价值版本的贝尔曼方程：

$$
Q^\pi(s,a)=\bar r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)
\sum_{a'\in\mathcal A(s')}\pi(a'\mid s')Q^\pi(s',a').
$$

$V^\pi$ 评价“处在这里怎么样”，$Q^\pi$ 评价“在这里先做这个动作怎么样”。两者都包含未来收益；$Q^\pi$ 不是即时奖励。它尤其适合比较同一状态下的不同动作。

#### 2.6 贯穿算例：评价随机策略

继续使用第 1.8 节的环境与各 $1/2$ 随机策略。简写 $V_1=V^\pi(s_1)$、$V_2=V^\pi(s_2)$，有

$$
\begin{aligned}
V_2&=2+0.9V_2,\\
V_1&=\tfrac12(1+0.9V_1)+\tfrac12(0+0.9V_2).
\end{aligned}
$$

先求出 $V_2=20$，代入得

$$
0.55V_1=9.5,\qquad V_1=\frac{190}{11}\approx17.2727.
$$

对应的矩阵方程是

$$
P^\pi=\begin{pmatrix}1/2&1/2\\0&1\end{pmatrix},\qquad
\mathbf r^\pi=\begin{pmatrix}1/2\\2\end{pmatrix},\qquad
\begin{pmatrix}11/20&-9/20\\0&1/10\end{pmatrix}
\begin{pmatrix}V_1\\V_2\end{pmatrix}
=\begin{pmatrix}1/2\\2\end{pmatrix}.
$$

再计算动作价值：

$$
\begin{aligned}
Q^\pi(s_1,a_{\mathrm{stay}})&=1+0.9\frac{190}{11}=\frac{182}{11}\approx16.5455,\\
Q^\pi(s_1,a_{\mathrm{up}})&=0+0.9\times20=18.
\end{aligned}
$$

验证：$\tfrac12\times182/11+\tfrac12\times18=190/11=V_1$。

注意，第一个动作价值不是一直留下的回报 $10$，因为它只规定第一步留下；以后仍按原随机策略行动，还有机会升级。升级动作的价值更高，提示我们可以改善原策略，但“最优”还需要第三章的论证。

### Chap 3: Bellman Optimality Equation

#### 3.1 最优价值与最优策略（Optimal Value and Policy）

对两个策略，若对所有状态都有 $V^{\pi_1}(s)\geq V^{\pi_2}(s)$，则称 $\pi_1$ 不差于 $\pi_2$。这是一种逐状态的偏序：两个策略也可能各自在不同状态更好，不能简单排出总顺序。

最优价值定义为

$$
V^*(s)=\sup_\pi V^\pi(s),\qquad Q^*(s,a)=\sup_\pi Q^\pi(s,a).
$$

在本文的有限折扣设定下，存在一个平稳确定性策略 $\pi^*$，同时对所有状态达到最优值：

$$
V^{\pi^*}(s)=V^*(s),\qquad\forall s\in\mathcal S.
$$

因此这里的上确界可以达到，可写成最大值。先用 $\sup$ 定义，再证明存在性，可以避免把“每个状态各有一个最优选择”未经证明就当成“同一个策略对所有状态最优”。

#### 3.2 从策略平均到动作最大化

给定一个候选未来价值函数 $v$，定义一次动作的前瞻值

$$
b_v(s,a)=\bar r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)v(s').
$$

若按概率分布 $\pi(\cdot\mid s)$ 选择动作，得到加权平均 $\sum_a\pi(a\mid s)b_v(s,a)$。任何加权平均都不超过其中最大项，而把全部概率放在一个最大项上就能取到上界。因此

$$
\max_{\pi(\cdot\mid s)}\sum_a\pi(a\mid s)b_v(s,a)
=\max_{a\in\mathcal A(s)}b_v(s,a).
$$

当未来也按最优方式行动时，得到贝尔曼最优方程（Bellman Optimality Equation，BOE）：

$$
\boxed{
V^*(s)=\max_{a\in\mathcal A(s)}
\left[\bar r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V^*(s')\right].
}
$$

这里最大化的是“即时奖励加未来价值”，不是仅最大化 $\bar r(s,a)$。环境转移依然随机，因此下一状态上的求和不能换成最大值；智能体能选择动作，不能选择环境随机结果。

#### 3.3 最优动作价值与贪心策略

最优状态价值和动作价值满足

$$
Q^*(s,a)=\bar r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V^*(s'),\qquad
V^*(s)=\max_aQ^*(s,a).
$$

代入得到动作价值形式的 BOE：

$$
\boxed{
Q^*(s,a)=\bar r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)
\max_{a'\in\mathcal A(s')}Q^*(s',a').
}
$$

注意运算顺序：先到达并观察下一状态 $s'$，再为这个状态选择最好的 $a'$，所以最大化在对 $s'$ 的求和内部。

已知 $Q^*$ 时，可在每个状态任取

$$
a^*(s)\in\operatorname*{arg\,max}_{a\in\mathcal A(s)}Q^*(s,a)
$$

并令 $\pi^*(a^*(s)\mid s)=1$。这称为对最优动作价值的贪心策略。“贪心”在这里并不短视，因为 $Q^*$ 本身已经考虑全部折扣未来。

#### 3.4 最优算子、非线性与压缩性

定义贝尔曼最优算子

$$
(T^*v)(s)=\max_a\left[\bar r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)v(s')\right].
$$

于是 BOE 为 $V^*=T^*V^*$。矩阵记法为

$$
\mathbf v^*=\max_\pi\{\mathbf r^\pi+\gamma P^\pi\mathbf v^*\},
$$

其中最大值逐分量理解，各状态分别选择动作。与固定策略的线性方程不同，最大动作可能随 $v$ 改变，故 $T^*$ 一般是非线性的，不能在求解前就假定一个已知的最优转移矩阵。

证明它是压缩映射，只需用到

$$
\left|\max_a x_a-\max_a y_a\right|\leq\max_a|x_a-y_a|.
$$

对任意 $v,w$ 和状态 $s$，

$$
\begin{aligned}
|(T^*v)(s)-(T^*w)(s)|
&\leq\gamma\max_a\left|\sum_{s'}p(s'\mid s,a)[v(s')-w(s')]\right|\\
&\leq\gamma\max_a\sum_{s'}p(s'\mid s,a)\|v-w\|_\infty\\
&=\gamma\|v-w\|_\infty.
\end{aligned}
$$

因此

$$
\boxed{\|T^*v-T^*w\|_\infty\leq\gamma\|v-w\|_\infty.}
$$

有限维空间 $\mathbb R^{|\mathcal S|}$ 在无穷范数下完备，压缩映射定理保证 $T^*$ 有唯一不动点 $u$，且 $v_{k+1}=T^*v_k$ 从任意初值收敛到 $u$。

#### 3.5 为什么这个唯一不动点就是最优价值

**第一步：它确实对应一个策略的价值。** 对不动点 $u$，在每个状态选择使 $(T^*u)(s)$ 达到最大值的动作，构造确定性策略 $\mu$。于是

$$
u=T^*u=T^\mu u.
$$

固定策略 $\mu$ 的贝尔曼方程只有一个解，所以 $u=V^\mu$。

**第二步：它不差于任何其他策略。** 对任意平稳策略 $\pi$，最大值不小于平均值，故

$$
u=T^*u\geq T^\pi u.
$$

算子 $T^\pi$ 保持逐分量大小关系，因为转移概率非负。反复应用它并取极限：

$$
u\geq T^\pi u\geq(T^\pi)^2u\geq\cdots
\longrightarrow V^\pi.
$$

因此 $u=V^\mu\geq V^\pi$，得证 $u=V^*$，且 $\mu$ 是最优策略。

即使允许策略依赖历史，上界仍成立：由每一步的 $u(s)\geq\bar r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)u(s')$，对任意策略连续展开 $N$ 步，可得

$$
u(s)\geq\mathbb E\left[\sum_{k=0}^{N-1}\gamma^kR_{t+k+1}+\gamma^Nu(S_{t+N})\mid S_t=s\right].
$$

由于 $u$ 有界且 $\gamma<1$，最后一项随 $N\to\infty$ 消失。因此历史依赖策略也不能超过 $u$；平稳确定性策略已经足够达到最优。

由此要区分：**最优价值函数唯一，但最优策略不一定唯一。** 如果某状态有多个动作并列最大，任选其一都可以；在并列最大动作之间随机化也可以最优。存在确定性最优策略，不等于所有最优策略都必须确定。

对于近似解，压缩性还给出实用的残差界：

$$
\|v-V^*\|_\infty\leq\frac{\|T^*v-v\|_\infty}{1-\gamma}.
$$

证明是在 $\|v-V^*\|_\infty$ 中加减 $T^*v$，使用三角不等式与压缩性后移项。它说明当 $\gamma$ 接近 $1$ 时，同样小的方程残差未必意味着同样小的价值误差。

#### 3.6 贯穿算例：求出并验证最优策略

最优方程为

$$
V_2^*=2+0.9V_2^*,\qquad
V_1^*=\max\{1+0.9V_1^*,\ 0.9V_2^*\}.
$$

先得 $V_2^*=20$。假设在 $s_1$ 选择升级，则 $V_1^*=18$；代回两个候选项：

$$
1+0.9\times18=17.2<18=0.9\times20.
$$

最大值确实由升级取得，候选解自洽；由唯一性，它就是最优解：

$$
\boxed{V^*(s_1)=18,\qquad V^*(s_2)=20.}
$$

最优策略是在 $s_1$ 立即升级，在 $s_2$ 持续获利。相应的 $Q^*(s_1,a_{\mathrm{stay}})=17.2$ 表示“现在先留下，下一步起采取最优策略”，而不是永远留下。

对比三个策略在 $s_1$ 的价值：永远留下为 $10$；原随机策略为 $190/11\approx17.2727$；立即升级为 $18$。这把“单步奖励”“策略评价”和“最优控制”联系起来。

#### 3.7 什么会改变最优策略

最优策略由奖励、状态转移和折扣因子共同决定。仍看上述例子，但允许 $\gamma$ 改变。两个确定性策略在 $s_1$ 的价值分别为

$$
V^{\mathrm{stay}}(s_1)=\frac{1}{1-\gamma},\qquad
V^{\mathrm{up}}(s_1)=\frac{2\gamma}{1-\gamma}.
$$

由于本例只有这两个平稳确定性策略，且至少有一个这样的策略最优，因此比较二者即可：

- $\gamma<1/2$：一直留下最优。
- $\gamma>1/2$：立即升级最优。
- $\gamma=1/2$：两动作并列最优，在它们之间随机选择也可最优。

奖励变换也要注意适用条件。若在无限时域折扣模型中，对**每一步的所有奖励**都作同一变换 $R'_t=\alpha R_t+\beta$，其中 $\alpha>0$，则

$$
G'_t=\alpha G_t+\frac{\beta}{1-\gamma},\qquad
V'^\pi(s)=\alpha V^\pi(s)+\frac{\beta}{1-\gamma}.
$$

所有策略价值同向缩放并平移，所以最优策略集合不变。但如果回合在终止后停止计分，只给终止前的奖励加常数，那么新增项取决于回合长度，可能改变最优策略。例如每步额外扣分可能鼓励尽早结束。不能把这两种设定混为一谈。

#### 3.8 前三章总结与自检

- **基础概念**：$(\mathcal S,\mathcal A,p,\gamma)$ 描述环境和目标，$\pi$ 描述行为，$G_t$ 描述一次未来收益。
- **策略评价**：给定 $\pi$，求 $V^\pi=T^\pi V^\pi$；动作按策略取平均，形成线性方程组。
- **最优性刻画**：求 $V^*=T^*V^*$；动作取最大值，形成非线性但具有压缩性的方程。
- **价值到策略**：已知 $Q^*$ 后，逐状态选择最大动作即可得到一个最优策略。

贝尔曼方程与贝尔曼最优方程是数学关系，不等于完整的学习算法。后续章节才讨论如何通过模型计算或交互样本逼近这些量；本笔记暂不展开值迭代、策略迭代、蒙特卡洛或时序差分算法。

读完后，应能独立回答：

1. 为什么状态必须保留预测未来所需的信息？策略随机与环境随机有什么区别？
2. 能否从 $G_t=R_{t+1}+\gamma G_{t+1}$ 推出贝尔曼方程，并解释每个求和的含义？
3. 为什么 $I-\gamma P^\pi$ 可逆？当 $\gamma=1$ 时，原证明哪里失效？
4. 为什么 $Q^\pi(s,a)$ 的第一步动作可以不由 $\pi$ 选择？
5. 最优方程中为什么对动作取最大值，但仍对下一状态取期望？
6. 为什么最优价值唯一却可能有多个最优策略？

#### 参考资料

章节顺序与核心主题对照赵世钰（Shiyu Zhao）的官方教材及课程资料；上面的两状态算例用于串联推导。

- [作者公开课主页](https://www.shiyuzhao.net/opencourse)：课程与教材入口。
- [官方教材第 1 章 Basic Concepts](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%201%20Basic%20Concepts.pdf)：状态、动作、策略、奖励、回报与 MDP。
- [官方教材第 2 章 State Values and Bellman Equation](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%202%20State%20Values%20and%20Bellman%20Equation.pdf)：价值、贝尔曼方程、矩阵形式与动作价值。
- [官方教材第 3 章 Optimal State Values and Bellman Optimality Equation](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%203%20Optimal%20State%20Values%20and%20Bellman%20Optimality%20Equation.pdf)：最优方程、压缩性、最优策略及奖励变换。
