# PaddleScience PINN 实现分析与优化总结

> 基于源码：https://github.com/PaddlePaddle/PaddleScience (develop 分支)
> 分析日期：2026-07-14
> 目标：提取可在 PyTorch 中复用的架构设计、优化技巧和最佳实践

---

## 目录

1. [整体架构](#1-整体架构)
2. [损失函数设计](#2-损失函数设计)
3. [网络架构](#3-网络架构)
4. [采样策略](#4-采样策略)
5. [训练策略](#5-训练策略)
6. [特殊优化技巧](#6-特殊优化技巧)
7. [PyTorch 迁移建议清单](#7-pytorch-迁移建议清单)

---

## 1. 整体架构

PaddleScience 的 PINN 实现遵循"约束（Constraint）+ 方程（PDE）+ 求解器（Solver）"的三层架构：

```
┌─────────────────────────────────────────────────────┐
│  Solver (ppsci/solver/solver.py)                    │
│  - 管理训练/评估/导出流程                            │
│  - 自动将 sympy 表达式转为可调用函数 (lambdify)       │
│  - 支持混合精度 (AMP)、分布式训练                      │
│  - 支持 EMA/SWA 模型平均                              │
│  - 可插拔 MTL loss aggregator                         │
├─────────────────────────────────────────────────────┤
│  Constraint (ppsci/constraint/)                     │
│  - InteriorConstraint: PDE 残差约束                   │
│  - BoundaryConstraint: 边界条件约束                    │
│  - InitialConstraint: 初始条件约束                     │
│  - SupervisedConstraint: 监督数据约束                  │
│  - PeriodicConstraint: 周期性边界条件                   │
│  - IntegralConstraint: 积分约束                        │
├─────────────────────────────────────────────────────┤
│  PDE Equation (ppsci/equation/pde/)                  │
│  - 用 sympy 符号化定义 PDE 方程                        │
│  - 自动编译为 callable 函数                           │
│  - 支持可学习参数 (learnable_parameters)               │
│  - 支持 detach_keys 防止梯度回传                       │
├─────────────────────────────────────────────────────┤
│  Geometry (ppsci/geometry/)                          │
│  - Interval/Rectangle/Cuboid/Triangle/Disk/Sphere/..│
│  - 多种采样: pseudo/Halton/LHS/uniform                │
│  - SDF (Signed Distance Function) 计算                │
├─────────────────────────────────────────────────────┤
│  Loss (ppsci/loss/)                                  │
│  - MSELoss / CausalMSELoss / PeriodicMSELoss         │
│  - MSELossWithL2Decay / FunctionalLoss               │
│  - MTL: Sum / GradNorm / AGDA / PCGrad / NTK / Relobralo │
└─────────────────────────────────────────────────────┘
```

**关键文件路径**：
- `ppsci/solver/solver.py` — 主求解器
- `ppsci/solver/train.py` — 训练循环
- `ppsci/constraint/` — 约束定义（interior/boundary/initial/supervised）
- `ppsci/loss/mse.py` — 损失函数，含 CausalMSELoss
- `ppsci/loss/mtl/` — 多任务学习损失聚合器
- `ppsci/equation/pde/base.py` — PDE 基类
- `ppsci/geometry/sampler.py` — 采样方法

---

## 2. 损失函数设计

### 2.1 基础损失：MSELoss

位置：`ppsci/loss/mse.py` — `MSELoss` 类

核心实现：

```python
def forward(self, output_dict, label_dict, weight_dict=None):
    losses = {}
    for key in label_dict:
        loss = F.mse_loss(output_dict[key], label_dict[key], "none")
        if weight_dict and key in weight_dict:
            loss *= weight_dict[key]        # 每个 batch 内的逐点加权
        if "area" in output_dict:
            loss *= output_dict["area"]     # 面积加权（几何边界）
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        if isinstance(self.weight, (float, int)):
            loss *= self.weight             # 全局 loss 级别权重
        elif isinstance(self.weight, dict) and key in self.weight:
            loss *= self.weight[key]        # 每个 output key 独立权重
        losses[key] = loss
    return losses
```

**设计要点**：
1. **逐点加权** (weight_dict) — 每个数据点可配不同权重，例如边界点可加高权重
2. **面积加权** (output_dict["area"]) — 不规则几何的边界和面积加权
3. **两级权重** — 全局 weight 和每个 key 的 weight
4. **返回 dict 而非标量** — 每个 loss 分量独立，以便 MTL Aggregator 合并

### 2.2 因果加权损失 (CausalMSELoss)

位置：`ppsci/loss/mse.py` — `CausalMSELoss` 类

实现因果 PINN 的时序因果加权：

```python
# 因果加权核心
loss_t = loss.reshape([self.n_chunks, -1])  # n_chunks x points_per_chunk
weight_t = paddle.exp(
    -self.tol * (self.acc_mat @ loss_t.mean(-1, keepdim=True))
)  # [n_chunks, 1]
loss = loss_t * weight_t.detach()
```

其中 `acc_mat` 是下三角矩阵，实现"只有前 k-1 个时间窗的损失很小时，第 k 个时间窗才有高权重"。

**引用**：[Wang et al., "Respecting causality in PINNs"](https://arxiv.org/abs/2203.07404)

### 2.3 带 L2 正则化的 MSE

位置：`ppsci/loss/mse.py` — `MSELossWithL2Decay` 类

对特定输出 key 施加 L2 正则化，控制网络输出的幅度。

### 2.4 周期性边界 MSE (PeriodicMSELoss)

位置：`ppsci/loss/mse.py` — `PeriodicMSELoss` 类

计算 batch 前半和后半的 MSE，用于强制 `u(x_left) == u(x_right)`。

### 2.5 函数式损失 (FunctionalLoss)

位置：`ppsci/loss/func.py`

允许用户自定义任意 loss 函数。XPINN 示例用此实现复杂的多子域损失组合：
- 源码：`examples/xpinn/xpinn.py`, `loss_fun` 函数（含 3 个子域 + 2 个界面）

### 2.6 损失权重配置模式

**示例**：`examples/heat_pinn/heat_pinn.py`
```python
bc_top = ppsci.constraint.BoundaryConstraint(
    ..., weight_dict={"u": cfg.TRAIN.weight.bc_top}, ...)
```

每个边界条件独立配置权重，通过 YAML 配置动态调节。

---

## 3. 网络架构

### 3.1 MLP 基础网络

位置：`ppsci/arch/mlp.py` — `MLP` 类

```python
class MLP(base.Arch):
    def __init__(self, input_keys, output_keys, num_layers, hidden_size,
                 activation="tanh", skip_connection=False, weight_norm=False,
                 periods=None, fourier=None, random_weight=None):
```

**特性**：
- **激活函数全面**：tanh、sin、cos、siren、stan（自缩放 tanh）、swish、gelu、relu 等
  - 位置：`ppsci/arch/activation.py`
- **权重归一化** (WeightNormLinear) — 分解为 `weight = g * v / ||v||`，稳定训练
- **随机权重分解** (RandomWeightFactorization) — 可学习参数分解，提升收敛
- **Skip Connection** — 偶数层跳跃连接
- **Period Embedding** — 周期性输入的 sin/cos 编码
- **Fourier Feature Embedding** — 随机傅里叶特征（RFF），提升高频学习能力

### 3.2 Modified MLP（改进 MLP）

位置：`ppsci/arch/mlp.py` — `ModifiedMLP` 类

关键改进（参考 SIREN 论文）：

```python
# forward_tensor 核心
u = self.embed_u(x)
v = self.embed_v(x)
y = x
for i, linear in enumerate(self.linears):
    y = linear(y)
    y = self.acts[i](y)
    y = y * u + (1 - y) * v  # 线性插值门控
```

**效果**：通过 `y = y * u + (1-y) * v` 门控机制缓解梯度路径问题。

### 3.3 SPINN（可分离 PINN）

位置：`ppsci/arch/spinn.py` — `SPINN` 类

核心思想：将每个维度用独立的 MLP 编码，然后做张量收缩：

```python
# 每个输入维度独立 MLP
self.branch_nets = nn.LayerList()
for i in range(input_dim):
    self.branch_nets.append(ModifiedMLP(input_keys[i], ...))

# 张量收缩（外积风格）
def forward_tensor(self, *xs):
    feature_f = [self.branch_nets[i](xs[i]) for i]
    output = []
    for key in output_keys:
        output_i = feature_f[0]  # shape: [B, r]
        for j in range(1, len(input_keys)):
            output_i = self._tensor_contraction(output_i, feature_f[j])
        output_i = output_i.sum(-1, keepdim=True)
        output.append(output_i)
    return output
```

**优势**：输入维度解耦，支持任意分辨率网格的推理（输入各维 batch_size 可不同），计算复杂度从 O(N^d) 降到 O(N*d)。

### 3.4 PIRBN（物理信息径向基网络）

位置：`jointContribution/PIRBN/pirbn.py`

使用 RBN（径向基网络）替代 MLP，并显式计算 NTK（神经正切核）进行自适应权重调整：

```python
class PIRBN(paddle.nn.Layer):
    def cal_ntk(self, x):
        # 计算域内点和边界点的 NTK 特征值
        lambda_g, lambda_b, Kg  # 用于自适应加权
```

### 3.5 PhyGeoNet（CNN + 几何变换）

位置：`examples/phygeonet/heat_equation.py`

使用 USCNN 架构，通过 body-fitted 网格变换将物理域映射到计算域，用 CNN 替代 MLP 加速：

```python
# 通过雅可比矩阵将物理坐标变换到计算域
dvdx = utils.dfdx(output_v, dydeta, dydxi, jinv)
d2vdx2 = utils.dfdx(dvdx, dydeta, dydxi, jinv)
```

### 3.6 输入/输出变换机制

位置：`ppsci/arch/base.py`

每个 Arch 基类支持：
- `register_input_transform(func)` — 对输入做预处理（如归一化、特征嵌入）
- `register_output_transform(func)` — 对输出做后处理（如硬约束边界条件）

**gPINN 示例**：`examples/gpinn/poisson_1d.py`
```python
def output_transform(in_, out):
    x = in_[invar]  # 输入 x
    u = out[outvar]  # 网络原始输出
    return {outvar: x + paddle.tanh(x) * paddle.tanh(np.pi - x) * u}
```
通过 `x + tanh(x)*tanh(π-x)*u` 硬编码满足边界条件 `u(0)=0, u(π)=π`。

### 3.7 可学习参数 PDE

位置：`ppsci/equation/pde/base.py` — `PDE.learnable_parameters`

PDE 类可以包含 `nn.ParameterList` 作为可学习参数，使得方程中的物理参数（如粘度、扩散系数）可被反向传播同时更新。

---

## 4. 采样策略

### 4.1 几何采样方法

位置：`ppsci/geometry/sampler.py`, `ppsci/geometry/geometry.py`

支持三种随机采样方法：
1. **`"pseudo"`** — 均匀随机采样（默认）
2. **`"Halton"`** — Halton 序列（低差异序列，更均匀覆盖）
3. **`"LHS"`** — 拉丁超立方采样（Latin Hypercube Sampling）

调用方式：
```python
geom.sample_interior(n, random="LHS", criteria=..., evenly=False)
geom.sample_boundary(n, random="pseudo", criteria=..., evenly=False)
```

### 4.2 `evenly=True` 均匀采样

位置：`ppsci/geometry/geometry.py` — `sample_interior` 方法

```python
if evenly:
    points = self.uniform_points(n)
```

用于热方程等示例中 PDE 残差点的均匀网格采样：
```python
# examples/heat_pinn/heat_pinn.py
pde_constraint = ppsci.constraint.InteriorConstraint(
    ..., evenly=True, name="EQ", ...)
```

### 4.3 criteria 筛选

每个采样接口支持 `criteria` 函数，用于只保留满足特定条件的点：

```python
# 仅保留 y=1 上的边界点
criteria=lambda x, y: np.isclose(y, 1)
```

底层采用重采样循环（while 循环），直至采满 n 个有效点。

### 4.4 Interval/Circle 等几何的均匀点采样

位置：`ppsci/geometry/geometry_1d.py`, `ppsci/geometry/geometry_2d.py`

几何类提供 `uniform_points(n, boundary)` 方法，在每个维度均匀分布点：
- Interval: `np.linspace(self.l, self.r, n)`
- Rectangle/Cuboid: 各维 linspace 的笛卡尔积

### 4.5 SDF 辅助采样

位置：`ppsci/geometry/geometry.py` — `sample_interior` 返回值带 `sdf` 字段

```python
sdf = -self.sdf_func(x)  # 取负使权重为正
```

SDF 值可以用于 loss 加权（边界附近采样点给更高权重），也可计算 SDF 导数做梯度增强。

### 4.6 XPINN 的子域采样策略

位置：`examples/xpinn/xpinn.py` — `train_dataset_transform_func`

在每个 epoch 的 DataLoader 中随机从子域和界面重采样：
```python
id_x1 = np.random.choice(_input["residual1_x"].shape[0],
                         cfg.MODEL.num_residual1_points, replace=False)
_input["residual1_x"] = _input["residual1_x"][id_x1, :]
```
每个子域独立配置采样点数，实现域分解采样。

### 4.7 SPINN 的每 epoch 重采样

位置：`examples/spinn/helmholtz3d.py` — `InteriorDataGenerator`

```python
class InteriorDataGenerator:
    def __call__(self):
        self.iter += 1
        if self.iter % 100 == 0:
            self._gen()  # 每 100 步重新采样
        return {...}
```
支持训练过程中定期重新采样，避免过拟合到特定采样点。

---

## 5. 训练策略

### 5.1 优化器支持

位置：`ppsci/optimizer/optimizer.py`

完整支持的优化器：
- **Adam**（默认首选，含 AMSGrad 变体）
- **AdamW**（带解耦权重衰减，支持无权重衰减参数白名单）
- **L-BFGS**（带 strong_wolfe 线搜索，常作为 Adam fine-tune 后段）
- **SGD**、**Momentum**、**RMSProp**
- **SOAP**（Shampoo + Adam，二阶优化器）
- **OptimizerList** — 多优化器组合（不同参数组用不同优化器）

**Adam + L-BFGS 两阶段训练模式**：
- 先 Adam 预训练，再 L-BFGS fine-tune
- Solver 自动检测 LBFGS 优化器并切换到 `train_LBFGS_epoch_func`
- L-BFGS 自动禁用 AMP

### 5.2 学习率调度

位置：`ppsci/optimizer/lr_scheduler.py`

支持的调度器：
- `ExponentialDecay` — 指数衰减（SPINN 示例使用）
- `CosineAnnealingDecay` — 余弦退火
- `StepDecay` — 阶梯衰减
- `ReduceOnPlateau` — 根据 loss 自适应
- 支持 `by_epoch` 和 `by_step` 两种模式

### 5.3 梯度累计

位置：`ppsci/solver/solver.py` — `update_freq` 参数

```python
if solver.update_freq > 1:
    total_loss = total_loss / solver.update_freq
# 每 update_freq 步进行一次 optimizer.step()
total_loss_scaled.backward()  # 每步都 backward
if iter_id % solver.update_freq == 0:
    solver.optimizer.step()    # 累计后更新
    solver.optimizer.clear_grad()
```

### 5.4 混合精度训练 (AMP)

位置：`ppsci/solver/solver.py` — `use_amp=True`, `amp_level="O1"`

```python
self.model, self.optimizer = amp.decorate(self.model, self.optimizer, self.amp_level)
```

限制：AGDA 和 PCGrad 不可用于 AMP。

### 5.5 模型平均

位置：`ppsci/utils/ema.py`

- **EMA** (Exponential Moving Average) — 指数移动平均
- **SWA** (Stochastic Weight Average) — 随机权重平均

### 5.6 梯度裁剪

全部优化器支持 `grad_clip` 参数：
```python
nn.ClipGradByNorm(clip_norm=1.0)   # 按范数裁剪
nn.ClipGradByValue(max=1.0)        # 按值裁剪
nn.ClipGradByGlobalNorm(1.0)       # 全局范数裁剪
```

---

## 6. 特殊优化技巧

### 6.1 多任务学习损失聚合 (MTL Loss Aggregator)

**这是 PINN 训练中最关键、最实用的优化技巧之一。**

位置：`ppsci/loss/mtl/`

#### 6.1.1 Sum（默认）— `mtl/sum.py`

简单求和。相当于传统 PINN 所有 loss 直接加。默认策略。

#### 6.1.2 GradNorm — `mtl/grad_norm.py`

**核心思想**：根据每个损失的梯度范数动态加权。

```python
weight_i = mean_grad_norm / grad_norm_i  # 梯度范数越大的 loss 权重越小
self.weight[i] = momentum * self.weight[i] + (1 - momentum) * weight_i
```

手动选取公式：
```
w_i^t = \frac{\overline{||∇_θ L_i^t||_2}}{||∇_θ L_i^t||_2}
```
- 使用动量平滑权重更新 (default: 0.9)
- 每 update_freq 步更新权重

#### 6.1.3 AGDA（自适应梯度下降算法）— `mtl/agda.py`

**核心思想**：用梯度冲突检测 + 梯度投影 + 自适应缩放。

适用于双任务（PDE loss + BC loss）场景。

流程：
1. 计算每个 loss 的梯度
2. 平滑后的 loss 比值 + 累计比值 → 调整权重缩放因子 ω
3. 计算 ω_f 和 ω_u 缩放梯度
4. 梯度投影：如果两个梯度的点积 < 0（冲突），减去冲突分量

引用：[Adaptive Gradient Descent Algorithm for PINN](https://pubs.aip.org/aip/pof/article-abstract/35/6/063608/2899773)

#### 6.1.4 PCGrad — `mtl/pcgrad.py`

**核心思想**：梯度冲突时投影到正交方向（与 AGDA 的梯度投影类似）。

#### 6.1.5 NTK 聚合 - `mtl/ntk.py`

基于神经正切核（NTK）理论，用 NTK 的特征值作为各 loss 的权重。

#### 6.1.6 Relobralo — `mtl/relobralo.py`

**核心思想**：ReLoBRaLO (Relative Loss Balancing with Random Lookback) — 基于训练过程中各 loss 下降速率的自适应权重。

### 6.2 Sympy → Callable 自动编译

位置：`ppsci/equation/pde/base.py`, `ppsci/utils/expression.py`

**关键设计**：

1. PDE 用 sympy 符号定义：
   ```python
   class Poisson(base.PDE):
       def __init__(self, dim):
           invars = self.create_symbols("x y z")[:dim]
           p = self.create_function("p", invars)
           poisson = sum(p.diff(invar, 2) for invar in invars)
           self.add_equation("poisson", poisson)
   ```

2. Solver 初始化时自动编译：
   ```python
   funcs = ppsci.lambdify(exprs, model, fuse_derivative=True)
   ```

3. Sympy 表达式和 callable 函数可混合使用，统一通过 `output_expr` 对外暴露

**`detach_keys` 机制**：在 sympy 层面指定哪些子项 `detach()`，防止不必要的梯度回传：
```python
# 在 NavierStokes 中 detach convection term
NS(..., detach_keys=("u", "v__y"))
```

### 6.3 硬约束边界条件

不限于 soft loss，支持硬编码边界条件：

**gPINN 示例** (`examples/gpinn/poisson_1d.py`)：
```python
output_transform = lambda in_, out: {
    outvar: x + paddle.tanh(x) * paddle.tanh(np.pi - x) * u
}
```

**PhyGeoNet 示例** (`examples/phygeonet/heat_equation.py`)：
```python
# 直接修改输出张量的边界值
output_v[:, 0, -pad_singleside:, ...] = 0  # 上边界
output_v[:, 0, :pad_singleside, ...] = 1    # 下边界
```

### 6.4 梯度增强 gPINN

位置：`examples/gpinn/poisson_1d.py`

```python
class gPINN1D(ppsci.equation.PDE):
    def __init__(self, invar, outvar):
        # 不仅约束 PDE 残差 (-dy_xx - f = 0)
        self.add_equation("res1", -dy_xx - f)
        # 还约束梯度 (-dy_xxx - df_x = 0)
        self.add_equation("res2", -dy_xxx - df_x)
```

在 InteriorConstraint 中同时添加两个方程：
```python
pde_constraint = ppsci.constraint.InteriorConstraint(
    equation["gPINN"].equations,
    {"res1": 0, "res2": 0},  # 两个残差别同时优化
    geom["line"], ...,
    ppsci.loss.MSELoss("mean", weight={"res2": 0.01}),  # 梯度项权重小
)
```

梯度增强项的 loss 权重通常设得比主 PDE 残差小（如 0.01），避免梯度主导训练。

### 6.5 域分解 XPINN

位置：`examples/xpinn/xpinn.py`, `examples/xpinn/model.py`

**核心架构**：
- 每个子域独立 MLP，共享输入但参数独立
- 界面一致性损失 (MSE_avg_q)：各子域在界面上的预测值应相等
- 界面残差一致性 (MSE_R)：界面两侧 PDE 残差应连续
- 每个界面点需要在相邻子域都 forward 一次

关键公式（来自 `_xpinn_loss`）：
```python
MSE_avg_q = mean(||u_q - u_avg||^2)     # 界面一致性
MSE_R = mean(||R(u_q) - R(u_neigh)||^2) # PDE 残差连续性
```

### 6.6 自适应采样

SPINN 示例 (`examples/spinn/helmholtz3d.py`)：
每 100 步重新采样训练点，避免过拟合：
```python
if self.iter % 100 == 0:
    self._gen()  # 重新采样 xc, yc, zc
```

XPINN 示例 (`examples/xpinn/xpinn.py`)：
每个 epoch 从原始数据集中随机子采样：
```python
id_x1 = np.random.choice(total_points, num_sample, replace=False)
```

### 6.7 SDF（符号距离函数）加速

位置：`ppsci/geometry/geometry.py` — `sample_interior`

采样结果自动包含 SDF 值：
```python
{**x_dict, **sdf_dict, **sdf_derives_dict}
```
- `sdf` — 每个点到边界的符号距离（取负后权重为正）
- `sdf__x`, `sdf__y` — SDF 相对输入的导数（可选）

**作用**：SDF 值可直接作为自适应权重，使接近边界的内部点获得更高权重。

### 6.8 Jacobian 计算缓存

位置：`ppsci/autodiff/ad.py`

```python
class _Jacobian:
    def __call__(self, i, j=None, ...):
        if i not in self.J:         # 缓存 J[i]
            self.J[i] = paddle.grad(y_i, xs, ...)
        return self.J[i]            # 复用缓存
```

`Jacobians` 类管理多级导数缓存。`clear()` 函数在每步 forward 后清空缓存，释放计算图。

### 6.9 显式梯度图形模式

位置：`examples/xpinn/xpinn.py`

```python
# 启用 PIR 原语以支持高阶导数
paddle.framework.core.set_prim_eager_enabled(True)
```

使用底层高阶导数 API 避免嵌套 `paddle.grad` 的计算图开销。

---

## 7. PyTorch 迁移建议清单

按优先级排序（P0 = 必须搬运，P1 = 强烈建议，P2 = 锦上添花）：

### P0（必须复现的基础架构）

| 编号 | 优化项 | 对应源码文件 | PyTorch 可行性 |
|------|--------|-------------|----------------|
| 1 | **Constraint 模式分离** — 将 PDE 残差、边界条件、初始条件、监督数据拆分为独立约束 | `ppsci/constraint/*.py` | ✅ 直接搬，`torch.utils.data.DataLoader` 即可 |
| 2 | **Sympy 符号化 PDE 定义** — 用 sympy 定义方程，自动解析求导 | `ppsci/equation/pde/base.py` | ✅ `sympy` 是跨框架的 |
| 3 | **自动微分类** — Jacobian 计算 + 缓存 + 多阶导数 | `ppsci/autodiff/ad.py` | ✅ 对应 `torch.autograd.grad` |
| 4 | **几何采样** — Interval/Rectangle/Cuboid + pseudo/Halton/LHS | `ppsci/geometry/*.py` | ✅ `scipy.stats.qmc.Halton`/`LatinHypercube` |
| 5 | **输入输出变换机制** — register_input_transform / register_output_transform | `ppsci/arch/base.py` | ✅ 纯 Python 函数包装 |
| 6 | **L-BFGS + Adam 两阶段训练** | `ppsci/optimizer/optimizer.py`, `ppsci/solver/train.py` | ✅ `torch.optim.LBFGS` 已完成 |

### P1（强烈建议）

| 编号 | 优化项 | 对应源码文件 | PyTorch 可行性 |
|------|--------|-------------|----------------|
| 7 | **GradNorm MTL 损失聚合** | `ppsci/loss/mtl/grad_norm.py` | ✅ 直接搬，只需把 `paddle.grad` → `torch.autograd.grad` |
| 8 | **CausalMSELoss** — 时序因果加权 | `ppsci/loss/mse.py` | ✅ 纯数学运算 |
| 9 | **ModifiedMLP（门控 MLP）** | `ppsci/arch/mlp.py` | ✅ 纯 PyTorch nn.Module |
| 10 | **权重归一化 (WeightNorm)** | `ppsci/arch/mlp.py` | ✅ `torch.nn.utils.weight_norm` |
| 11 | **硬约束边界条件** — output_transform | `examples/gpinn/poisson_1d.py` | ✅ 任意神经网络 forward hook |
| 12 | **detach_keys 机制** — 符号化 detach 子表达式 | `ppsci/equation/pde/base.py` | ✅ Sympy 层面处理后在代码中调用 `.detach()` |
| 13 | **逐点加权 loss** — weight_dict 机制 | `ppsci/loss/mse.py` | ✅ 直接搬 |
| 14 | **每步重采样** — epoch 内定期更新采样点 | `examples/spinn/helmholtz3d.py` | ✅ 自定义 Dataset |

### P2（锦上添花，按需添加）

| 编号 | 优化项 | 对应源码文件 | PyTorch 可行性 |
|------|--------|-------------|----------------|
| 15 | **SPINN 可分离网络** | `ppsci/arch/spinn.py` | ✅ 跨框架思路，实现独立 |
| 16 | **AGDA 自适应梯度** | `ppsci/loss/mtl/agda.py` | ⚠️ 需要梯度投影，API 不同但可实现 |
| 17 | **PCGrad 梯度投射** | `ppsci/loss/mtl/pcgrad.py` | ⚠️ 同上 |
| 18 | **Fourier Feature Embedding（RFF）** | `ppsci/arch/mlp.py` | ✅ `torch.nn.Linear` + 固定随机种子 |
| 19 | **Period Embedding** | `ppsci/arch/mlp.py` | ✅ 直接搬 |
| 20 | **Siren 初始化 + sin 激活** | `ppsci/arch/activation.py` | ✅ SIREN 方法 |
| 21 | **Stan 激活（自缩放 tanh）** | `ppsci/arch/activation.py` | ✅ `tanh(x)*(1+beta*x)` |
| 22 | **NTK 自适应权重** | `ppsci/loss/mtl/ntk.py` | ⚠️ 计算开销大，小模型可 |
| 23 | **Relobralo 自适应权重** | `ppsci/loss/mtl/relobralo.py` | ✅ 纯数学运算 |
| 24 | **EMA/SWA 模型平均** | `ppsci/utils/ema.py` | ✅ `torch.optim.swa_utils` |
| 25 | **梯度裁剪** | `ppsci/optimizer/optimizer.py` | ✅ `torch.nn.utils.clip_grad_norm_` |
| 26 | **混合精度 (AMP)** | `ppsci/solver/solver.py` | ✅ `torch.cuda.amp` |
| 27 | **梯度累计** | `ppsci/solver/train.py` | ✅ 手动实现 `loss/update_freq` |

### 关键 PyTorch API 映射表

| Paddle API | PyTorch API | 说明 |
|-----------|-------------|------|
| `paddle.grad` | `torch.autograd.grad` | 自动求导 |
| `paddle.nn.Layer` | `torch.nn.Module` | 基类 |
| `paddle.nn.Linear` | `torch.nn.Linear` | 线性层 |
| `paddle.nn.functional.mse_loss` | `torch.nn.functional.mse_loss` | MSE 损失 |
| `paddle.nn.LayerList` | `torch.nn.ModuleList` | 模块列表 |
| `self.create_parameter` | `torch.nn.Parameter` | 可学习参数 |
| `self.register_buffer` | `self.register_buffer` | 缓冲张量 |
| `paddle.linalg.norm` | `torch.linalg.norm` | 向量/矩阵范数 |
| `paddle.concat` | `torch.cat` | 拼接 |
| `paddle.exp` | `torch.exp` | 指数运算 |
| `paddle.tril` | `torch.tril` | 下三角矩阵 |
| `set_value` | `.data.copy_` | 原地赋值 |
| `global_step` | 自增计数器 | 训练步数 |

---

## 总结

PaddleScience 的 PINN 实现主要有以下突出设计：

1. **约束解耦** — 将不同类型的物理约束（PDE/BC/IC/Data）分别定义为独立模块，通过 Solver 统一管理，天然支持任意组合
2. **符号微分 + 自动编译** — 用 sympy 定义 PDE 后自动编译为 callable，用户无需手写导数计算
3. **MTL 损失聚合** — GradNorm/AGDA/PCGrad/Relobralo/NTK 等动态权重策略解决了 PINN 的梯度不平衡问题
4. **网络灵活** — MLP/ModifiedMLP/SPINN/CNN/KAN 多种架构，支持 weight norm、skip connection、period embedding 等增强技巧
5. **采样多样化** — LHS/Halton/均匀/SDF 加权，支持周期性重采样和自适应采样
6. **因果损失** — 显式实现 Causality 加权损失，解决时间相关 PDE 的演进顺序问题

这些设计绝大部分可以直接迁移到 PyTorch，仅需将 Paddle API 替换为对应的 PyTorch API。
