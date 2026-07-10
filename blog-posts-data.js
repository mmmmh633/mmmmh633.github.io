// Auto-generated blog posts data
// Generated: 2026-07-10
const blogPosts = [

  // ===== 1. PaddleScience 调研报告 =====
  {
    id: "paddlescience-survey",
    title: "PaddleScience 调研报告",
    date: "2026-07-06",
    category: "调研",
    excerpt: "百度飞桨生态下的科学计算套件调研，覆盖架构、能力、案例及与 NVIDIA Modulus、DeepXDE 的对比分析。",
    content: `<blockquote><p>调研日期：2026-07-06 | 调研人：秘书</p></blockquote>
<hr>
<h2>一、概述</h2>
<p>PaddleScience 是百度飞桨（PaddlePaddle）生态下的科学计算套件，利用深度神经网络的学习能力和飞桨框架的自动（高阶）微分机制，解决物理、化学、气象等领域的 PDE/ODE 问题。项目由百度飞桨团队维护，采用 Apache 2.0 开源协议。</p>
<ul>
<li><strong>定位</strong>：国产深度学习驱动的科学计算框架，对标 NVIDIA Modulus、DeepXDE</li>
<li><strong>GitHub</strong>：<a href="https://github.com/PaddlePaddle/PaddleScience" target="_blank">PaddlePaddle/PaddleScience</a></li>
<li><strong>文档</strong>：<a href="https://paddlescience-docs.readthedocs.io/zh-cn/latest/" target="_blank">paddlescience-docs.readthedocs.io</a></li>
<li><strong>PyPI</strong>：<code>pip install paddlesci</code></li>
<li><strong>Conda</strong>：<code>conda install -c paddlescience paddlescience</code></li>
<li><strong>许可证</strong>：Apache 2.0</li>
</ul>
<hr>
<h2>二、主要特点</h2>
<h3>2.1 三种求解范式</h3>
<table>
<tr><th>范式</th><th>说明</th><th>适用场景</th></tr>
<tr><td>物理机理驱动</td><td>PDE 作为损失函数约束，无监督/半监督</td><td>边界条件明确，监督数据稀缺</td></tr>
<tr><td>数据驱动</td><td>纯监督学习，从数据中学习映射</td><td>有大量模拟/实验数据</td></tr>
<tr><td>数理融合</td><td>同时使用 PDE 约束 + 数据约束</td><td>部分物理规律已知，部分数据可用</td></tr>
</table>
<h3>2.2 架构模块（12 个核心模块）</h3>
<ul>
<li><strong>Arch</strong>：神经网络模型（MLP、SPINN、Transformer、GNN 等）</li>
<li><strong>AutoDiff</strong>：高阶自动微分（Jacobi、Hessian）</li>
<li><strong>Constraint</strong>：统一体素——将 PDE 约束、边界约束、监督约束统一为"约束"抽象，使机理驱动和数据驱动共享训练流程</li>
<li><strong>Data</strong>：数据集、数据预处理与批处理</li>
<li><strong>Equation</strong>：基于 SymPy 的符号方程定义，支持联立方程组</li>
<li><strong>Geometry</strong>：几何采样，支持简单几何（矩形、圆）和复杂 STL 几何，支持布尔运算</li>
<li><strong>Loss</strong>：多种损失函数 + 多任务学习（MTL）权重策略</li>
<li><strong>Metric</strong>：L2、MSE 等评估指标</li>
<li><strong>Optimizer</strong>：优化器 + 学习率调度器</li>
<li><strong>Solver</strong>：全局调度器，负责训练、评估、可视化的流程编排</li>
<li><strong>Utils</strong>：日志、配置（Hydra）、可视化（Matplotlib/VTK）</li>
<li><strong>Experimental</strong>：实验性功能模块</li>
</ul>
<h3>2.3 关键能力</h3>
<ul>
<li><strong>符号方程</strong>：支持 SymPy 表达方程，自动推导 PDE 损失所需的微分项</li>
<li><strong>几何能力</strong>：STL 复杂几何加载、采样、布尔运算；支持 2D/3D</li>
<li><strong>边界条件</strong>：Dirichlet、Neumann、Robin + 自定义 BC</li>
<li><strong>实验管理</strong>：Hydra 配置文件驱动，支持实验源码跟踪、一键并行超参搜索</li>
<li><strong>可视化</strong>：Matplotlib 2D + VTK 3D 可视化输出</li>
<li><strong>多硬件</strong>：CPU、GPU（CUDA）、XPU（昆仑芯）等</li>
<li><strong>Type Hints</strong>：全库完善的类型标注</li>
<li><strong>AI Studio 集成</strong>：大部分案例可直接在百度 AI Studio 在线运行</li>
</ul>
<h3>2.4 案例覆盖</h3>
<p>案例按领域分为五大类，超过 60+ 个：</p>
<table>
<tr><th>领域</th><th>案例数</th><th>代表案例</th></tr>
<tr><td><strong>数学</strong></td><td>~17</td><td>Allen-Cahn、Helmholtz、DeepONet、Lorenz、XPINN、符号回归 Transformer4SR</td></tr>
<tr><td><strong>流体力学</strong></td><td>~15</td><td>圆柱绕流、方腔流、达西流、激波、涡激振动、气动外形设计、DrivAerNet</td></tr>
<tr><td><strong>结构力学</strong></td><td>~8</td><td>双调和方程、拓扑优化、支架/控制臂应力、EPNN</td></tr>
<tr><td><strong>传热</strong></td><td>~4</td><td>热交换器、Heat-PINN、PhyGeoNet、芯片散热</td></tr>
<tr><td><strong>地球科学</strong></td><td>~10+</td><td>FourCastNet、GraphCast、NowcastNet、气象大模型系列（FengWu、FuXi 等）</td></tr>
<tr><td><strong>材料科学</strong></td><td>~5</td><td>hPINNs、CGCNN、电池建模</td></tr>
<tr><td><strong>化学</strong></td><td>~4</td><td>SMC Reac、Moflow、分子生成</td></tr>
</table>
<hr>
<h2>三、技术路线对比</h2>
<h3>3.1 与同类型框架对比</h3>
<table>
<tr><th>维度</th><th>PaddleScience</th><th>NVIDIA Modulus</th><th>DeepXDE</th><th>PhysicsNemo</th></tr>
<tr><td><strong>底层框架</strong></td><td>PaddlePaddle</td><td>PyTorch</td><td>TensorFlow/PyTorch/JAX</td><td>PyTorch</td></tr>
<tr><td><strong>PINN 支持</strong></td><td>✅ 完善</td><td>✅ 完善</td><td>✅ 核心功能</td><td>✅ NVIDIA 生态整合</td></tr>
<tr><td><strong>数据驱动算子</strong></td><td>✅ Transformer/LNO/DeepONet</td><td>✅ FNO/AFNO/MeshGraphNet</td><td>⚠️ 基本</td><td>✅ FNO 系列</td></tr>
<tr><td><strong>几何处理</strong></td><td>✅ STL + SymPy 几何</td><td>✅ CSG + STL</td><td>⚠️ 简单几何</td><td>✅ 继承 Modulus</td></tr>
<tr><td><strong>符号方程</strong></td><td>✅ SymPy 原生</td><td>⚠️ 需手写</td><td>⚠️ 需手写</td><td>⚠️ 需手写</td></tr>
<tr><td><strong>多硬件</strong></td><td>CUDA/XPU(昆仑)/CPU</td><td>CUDA only</td><td>CPU/CUDA</td><td>CUDA only</td></tr>
<tr><td><strong>开源协议</strong></td><td>Apache 2.0</td><td>Apache 2.0</td><td>LGPL-2.1</td><td>Apache 2.0</td></tr>
<tr><td><strong>中文文档</strong></td><td>✅ 完善</td><td>⚠️ 英文为主</td><td>⚠️ 英文为主</td><td>⚠️ 英文为主</td></tr>
<tr><td><strong>社区生态</strong></td><td>百度飞桨生态+AI Studio</td><td>NVIDIA 官方</td><td>Lu Lu 学术维护</td><td>NVIDIA 官方</td></tr>
<tr><td><strong>工业部署</strong></td><td>Paddle Inference/Serving</td><td>TensorRT/Triton</td><td>学术为主</td><td>TensorRT/Triton</td></tr>
<tr><td><strong>最新 Star</strong></td><td>~1.5K (2026)</td><td>~12K+</td><td>~3.5K</td><td>Modulus 子项目</td></tr>
</table>
<h3>3.2 优劣势分析</h3>
<p><strong>PaddleScience 优势：</strong></p>
<ul>
<li>🇨🇳 <strong>中文文档与社区</strong>：国内开发者上手快，飞桨生态配套齐全（AI Studio 免费算力、飞桨企业版）</li>
<li>📐 <strong>符号方程抽象</strong>：唯一原生集成 SymPy 的框架，方程定义直观、可读性强</li>
<li>🔧 <strong>统一 Constraint 设计</strong>：物理约束和数据约束共用训练流程，数理融合场景极方便</li>
<li>🧪 <strong>实验工程化</strong>：Hydra 配置管理、实验追踪、并行搜索，适合大规模消融实验</li>
<li>🏭 <strong>国产化适配</strong>：昆仑芯 XPU 支持，适合国产硬件环境</li>
<li>📚 <strong>案例丰富</strong>：覆盖气象、流体、结构、材料、化学等多学科</li>
</ul>
<p><strong>PaddleScience 劣势：</strong></p>
<ul>
<li>🎯 <strong>框架依赖</strong>：必须使用 PaddlePaddle 生态，与 PyTorch 技术栈不兼容；习惯 PyTorch 的用户迁移成本高</li>
<li>📉 <strong>社区规模较小</strong>：GitHub Star 约 1.5K，相比 Modulus（12K+）社区活跃度和三方贡献差距明显</li>
<li>🔄 <strong>更新滞后</strong>：依赖 PaddlePaddle 发版节奏，部分前沿模型（如最新 Transformer 变体）收录可能滞后于 PyTorch 生态</li>
<li>🚫 <strong>NVIDIA 绑定弱</strong>：GPU 优化不如 Modulus 深度，TensorRT/Triton 部署路径不如 NVIDIA 生态顺畅</li>
<li>📝 <strong>学术引用少</strong>：相比 Modulus 和 DeepXDE 在顶会论文中的引用量，PaddleScience 的学术影响力还在积累阶段</li>
<li>🧱 <strong>复杂几何偏弱</strong>：STL 支持可用但不如 Modulus 的 CSG（Constructive Solid Geometry）灵活</li>
</ul>
<p><strong>选型建议：</strong></p>
<table>
<tr><th>场景</th><th>推荐</th></tr>
<tr><td>国内团队、中文环境、飞桨技术栈</td><td><strong>PaddleScience</strong> ✅</td></tr>
<tr><td>NVIDIA GPU 深度集成、工业部署</td><td>Modulus</td></tr>
<tr><td>快速原型验证、跨框架需要</td><td>DeepXDE</td></tr>
<tr><td>PyTorch + NVIDIA 全家桶</td><td>PhysicsNemo / Modulus</td></tr>
<tr><td>国产硬件（昆仑芯）</td><td><strong>PaddleScience</strong> ✅</td></tr>
<tr><td>PINN 学术研究</td><td>DeepXDE 或 PaddleScience 均可</td></tr>
</table>
<hr>
<h2>四、安装与入门</h2>
<h3>4.1 安装</h3>
<p><a href="https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html" target="_blank">PaddlePaddle 安装指南</a></p>
<pre><code>python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/</code></pre>
<h3>4.2 快速入门示例</h3>
<pre><code>import numpy as np
import ppsci
from ppsci.utils import logger

# 1. 设置随机种子
ppsci.utils.misc.set_random_seed(42)

# 2. 构建几何（-π 到 π）
geom = ppsci.geometry.Interval(-np.pi, np.pi)

# 3. 定义方程
def pde_func(out):
    return out["du/dx"] - ppsci.autodiff.cos(out["x"])

# 4. 构建约束
constraint = ppsci.constraint.InteriorConstraint(
    pde_func, {"x": geom},
    ...
)

# 5. 训练
model = ppsci.arch.MLP(("x",), ("u",), 5, 64)
solver = ppsci.solver.Solver(model, constraint)
solver.train()</code></pre>
<p>完整示例见官方文档：<a href="https://paddlescience-docs.readthedocs.io/zh-cn/latest/quickstart/" target="_blank">快速开始</a></p>
<hr>
<h2>五、学术成果</h2>
<p>PaddleScience 团队及相关合作者在以下顶会/期刊发表成果（部分）：</p>
<ol>
<li>Wang & Wang (2024) — <strong>Latent Neural Operator for Forward and Inverse PDE Problems</strong>，NeurIPS</li>
<li>Deng et al. (2024) — <strong>Geometry-Guided Conditional Adaption for Surrogate Models</strong>，IJCAI</li>
<li>Xu et al. (2024) — <strong>YingLong: Skillful High Resolution Regional Short Term Forecasting</strong>，arXiv</li>
<li>Chu et al. (2024) — <strong>Flow reconstruction over a SUBOFF model based on LBM + PINN</strong>，Ocean Engineering</li>
<li>Huang et al. (2024) — <strong>PINNs for advection-diffusion-Langmuir adsorption</strong>，Physics of Fluids</li>
<li>Lu et al. (2024) — <strong>Multi-Branch Physics-Informed deep operator neural network for thermal simulation</strong>，Physics of Fluids</li>
</ol>
<p>完整列表：<a href="https://paddlescience-docs.readthedocs.io/zh-cn/latest/academic/" target="_blank">学术成果页</a></p>
<hr>
<h2>六、参考链接</h2>
<table>
<tr><th>资源</th><th>链接</th></tr>
<tr><td>GitHub 仓库</td><td><a href="https://github.com/PaddlePaddle/PaddleScience" target="_blank">PaddlePaddle/PaddleScience</a></td></tr>
<tr><td>官方文档（中文）</td><td><a href="https://paddlescience-docs.readthedocs.io/zh-cn/latest/" target="_blank">paddlescience-docs</a></td></tr>
<tr><td>PyPI</td><td><a href="https://pypi.org/project/paddlesci/" target="_blank">paddlesci</a></td></tr>
<tr><td>AI Studio 实训社区</td><td><a href="https://aistudio.baidu.com/aistudio/projectoverview/public?topic=15" target="_blank">AI Studio</a></td></tr>
<tr><td>DeepWiki</td><td><a href="https://deepwiki.com/PaddlePaddle/PaddleScience" target="_blank">DeepWiki</a></td></tr>
<tr><td>百度飞桨官网</td><td><a href="https://www.paddlepaddle.org.cn/" target="_blank">飞桨</a></td></tr>
<tr><td>NVIDIA Modulus（对比参考）</td><td><a href="https://github.com/NVIDIA/modulus" target="_blank">Modulus</a></td></tr>
<tr><td>DeepXDE（对比参考）</td><td><a href="https://github.com/lululxvi/deepxde" target="_blank">DeepXDE</a></td></tr>
</table>
<hr>
<h2>七、总结</h2>
<p>PaddleScience 是国内目前最完整的深度学习科学计算开源框架，架构设计合理（Constraint 抽象 + SymPy 方程 + Geometry 模块），中文生态完善。对于建达所在团队（飞书生态、国产化需求、物理信息嵌入方向），PaddleScience 具有天然亲和力，可以直接作为 PINN 推进的技术选项之一。</p>
<p>核心取舍点：<strong>如果团队技术栈是 PaddlePaddle 或愿意切换，PaddleScience 是国产最优选；如果已有 PyTorch 沉淀，Modulus/PhysicsNemo 更顺滑。</strong></p>
<p>建议下一步：用 PaddleScience 跑一个 Allen-Cahn 或传热案例，直观感受开发效率，再做决策。</p>`
  },

  // ===== 2. 物理信息机器学习框架调研 =====
  {
    id: "physics-ml-frameworks-survey",
    title: "物理信息机器学习框架调研",
    date: "2026-07-06",
    category: "调研",
    excerpt: "整合 DeepXDE、PaddleScience、NVIDIA PhysicsNeMo 及 DMD/POD/DEIM 降维方法，全方位对比物理信息机器学习框架。",
    content: `<blockquote><p>整理自飞书知识库「物理信息机器学习」，整合 DeepXDE、PaddleScience、DMD/POD/DEIM 降维方法，补充 NVIDIA PhysicsNeMo。核实并标注要点。</p><p>整理时间：2026-07-06</p></blockquote>
<hr>
<h2>一、总体思路</h2>
<p>物理信息机器学习（Physics-Informed Machine Learning）的核心路线：将物理规律（PDE、边界条件、守恒律）嵌入神经网络训练过程，使模型在数据稀缺时仍能保持物理一致性。</p>
<p>当前主流范式：</p>
<ol>
<li><strong>物理信息驱动的 PINN</strong>：将 PDE 残差作为损失函数的一部分，无需标注数据即可求解</li>
<li><strong>数据驱动的神经算子</strong>：从仿真数据中学习 PDE 算子映射（如 DeepONet、FNO），训练后可即时推理</li>
<li><strong>数理融合</strong>：PINN + 数据监督的混合策略</li>
<li><strong>降维加速</strong>：POD/DMD 提取低维模态 + DEIM 加速非线性项，构建快速降阶模型（ROM）</li>
</ol>
<hr>
<h2>二、框架全景对比</h2>
<table>
<tr><th>维度</th><th>DeepXDE</th><th>PaddleScience</th><th>NVIDIA PhysicsNeMo</th></tr>
<tr><td><strong>开发方</strong></td><td>布朗大学 Lu Lu 课题组</td><td>百度</td><td>NVIDIA</td></tr>
<tr><td><strong>底层框架</strong></td><td>TF/PyTorch/JAX/Paddle 多后端</td><td>PaddlePaddle 专用</td><td>PyTorch</td></tr>
<tr><td><strong>开源协议</strong></td><td>LGPL-2.1</td><td>Apache 2.0</td><td>Apache 2.0</td></tr>
<tr><td><strong>核心能力</strong></td><td>PINN + DeepONet 算子学习</td><td>PINN + 数据驱动 + 算子学习</td><td>PINN + 神经算子 + GNN + 扩散模型</td></tr>
<tr><td><strong>PINN 支持</strong></td><td>✅ 核心功能</td><td>✅ 15 模块之一</td><td>✅ via physicsnemo.sym（SymPy PDE）</td></tr>
<tr><td><strong>3D 几何</strong></td><td>⚠️ 基础支持</td><td>✅ pymesh/open3d 完整支持</td><td>✅ 点云/mesh 数据管道</td></tr>
<tr><td><strong>分布式训练</strong></td><td>⚠️ utils 中有基础支持</td><td>❌ 单机为主</td><td>✅ 多 GPU/多节点原生支持</td></tr>
<tr><td><strong>预训练模型</strong></td><td>❌</td><td>❌</td><td>✅ NGC 模型库</td></tr>
<tr><td><strong>生态定位</strong></td><td>学术研究、算子学习</td><td>国产框架、工业验证</td><td>工业级数字孪生、大规模部署</td></tr>
</table>
<blockquote><p>✅ <strong>核实</strong>：DeepXDE 确实由布朗大学 Lu Lu（陆路）主导，是首个提出 DeepONet 的团队。多后端设计独特，但生态和案例丰富度不及 PaddleScience 和 PhysicsNeMo。</p></blockquote>
<hr>
<h2>三、DeepXDE</h2>
<p><img src="物理信息机器学习框架调研/DeepXDE_01.png" alt="DeepXDE 框架概览"></p>
<h3>3.1 概述</h3>
<p>DeepXDE 是布朗大学 Lu Lu 课题组开发维护的开源 AI4S 框架，主要支持 PINN 和算子学习（DeepONet）。由于作者本人是 DeepONet 的首个提出者，对这两类学习方程支持较为完善。</p>
<p><strong>特点：</strong></p>
<ul>
<li>支持 TensorFlow、PyTorch、JAX、PaddlePaddle 多后端</li>
<li>通过环境变量切换后端：<code>DDE_BACKEND=pytorch python pde.py</code></li>
<li>后端切换机制：在 <code>backend.py</code> 中定义空方法，运行时根据环境变量动态替换为对应框架实现</li>
<li>功能相对简洁，依赖库更常见，但三维问题处理和生态案例方面不如赛桨</li>
<li>对数据驱动的大模型缺乏支持，多后端下统一编写 AI 网络模型较困难</li>
</ul>
<blockquote><p>✅ <strong>核实</strong>：DeepXDE 多后端设计确实通过运行时动态导入实现。<code>load_backend()</code> 在导入时自动执行。该设计灵活但对新模型架构的支持有局限。</p></blockquote>
<p><img src="物理信息机器学习框架调研/DeepXDE_02.png" alt="DeepXDE 后端选择提示"></p>
<h3>3.2 模块结构</h3>
<table>
<tr><th>模块</th><th>功能</th></tr>
<tr><td><strong>backend</strong></td><td>多后端统一封装，运行时动态选择 TF/PyTorch/JAX/Paddle</td></tr>
<tr><td><strong>data</strong></td><td>数据处理、几何采样、损失计算</td></tr>
<tr><td><strong>geometry</strong></td><td>1D~ND 超几何、时间域、点云，支持并/交/差布尔运算</td></tr>
<tr><td><strong>gradients</strong></td><td>雅可比矩阵、海塞矩阵计算（前向/后向两种模式）</td></tr>
<tr><td><strong>icbc</strong></td><td>初始条件 + 边界条件：Dirichlet、Neumann、周期边界</td></tr>
<tr><td><strong>nn</strong></td><td>网络结构：FNN、PFNN、DeepONet、DeepONetCartesianProd、MIONetCartesianProd、PODDeepONet、PODMIONet</td></tr>
<tr><td><strong>optimizer</strong></td><td>L-BFGS 优化器（需要闭包 callback 模式）</td></tr>
<tr><td><strong>utils</strong></td><td>数据转换、标准化、多 GPU/多服务器训练辅助</td></tr>
<tr><td><strong>zcs</strong></td><td>零条件累积（Zero Conditional Sum），惰性求值策略优化梯度计算</td></tr>
</table>
<h3>3.3 L-BFGS 使用要点</h3>
<p>DeepXDE 的 L-BFGS 使用 Torch 原生接口，需要定义 <code>closure()</code> 函数：</p>
<pre><code>optimizer = optim.LBFGS([x], lr=1.0, max_iter=5, line_search_fn='strong_wolfe')

def closure():
    optimizer.zero_grad()
    loss = rosenbrock(x)
    loss.backward()
    return loss

for i in range(100):
    optimizer.step(closure)</code></pre>
<blockquote><p>✅ <strong>核实</strong>：L-BFGS 的 closure 模式是 PyTorch 标准用法，DeepXDE 封装正确。</p></blockquote>
<hr>
<h2>四、PaddleScience</h2>
<h3>4.1 概述</h3>
<p>PaddleScience 是百度基于 PaddlePaddle 的二次封装科学计算框架，通过 <code>arch</code>、<code>constraint</code>、<code>equation</code> 等模块简化了网络构造和 PDE 定义的难度。</p>
<p><img src="物理信息机器学习框架调研/Paddle_01.png" alt="赛桨结构及功能梳理"></p>
<p><img src="物理信息机器学习框架调研/Paddle_02.png" alt="赛桨改进方案"></p>
<p><strong>支持三种求解范式：</strong></p>
<ol>
<li>无监督 PINN（物理机理驱动）</li>
<li>数据驱动的监督学习</li>
<li>神经算子学习（DeepONet、FNO）</li>
</ol>
<blockquote><p>✅ <strong>核实</strong>：PaddleScience 的 15 模块架构设计合理，对前沿 AI4S 模型（如 Transformer、扩散模型）也进行了封装。框架鼓励开发者将文献中的模型用赛桨实现并贡献。</p></blockquote>
<p><strong>物理信息求解案例：</strong></p>
<table>
<tr><th>案例</th><th>结果截图</th></tr>
<tr><td>2D 定常方腔流</td><td><img src="物理信息机器学习框架调研/Paddle_04.png" alt="方腔流"></td></tr>
<tr><td>多孔介质流动</td><td><img src="物理信息机器学习框架调研/Paddle_05.png" alt="多孔介质"></td></tr>
<tr><td>管道流动</td><td><img src="物理信息机器学习框架调研/Paddle_07.png" alt="管道流动"></td></tr>
<tr><td>3D 血管仿真</td><td><img src="物理信息机器学习框架调研/Paddle_09.png" alt="3D血管"></td></tr>
<tr><td>瞬态方腔流动</td><td><img src="物理信息机器学习框架调研/Paddle_11.png" alt="瞬态方腔"></td></tr>
<tr><td>2D 平板变形</td><td><img src="物理信息机器学习框架调研/Paddle_13.png" alt="平板变形"></td></tr>
<tr><td>汽车控制臂变形</td><td><img src="物理信息机器学习框架调研/Paddle_15.png" alt="控制臂"></td></tr>
</table>
<h3>4.2 15 大模块</h3>
<p><img src="物理信息机器学习框架调研/Paddle_16.png" alt="PaddleScience 15 模块架构图"></p>
<table>
<tr><th>模块</th><th>功能</th></tr>
<tr><td><strong>arch</strong></td><td>网络构建：MLP、UNet、Transformer、DeepONet 等</td></tr>
<tr><td><strong>autodiff</strong></td><td>雅可比/海塞矩阵自动微分</td></tr>
<tr><td><strong>constraint</strong></td><td>PDE 内部约束、边界约束、积分约束、数据监督约束</td></tr>
<tr><td><strong>data</strong></td><td>CSV/VTU/NPZ/MAT 等多格式数据加载与变换</td></tr>
<tr><td><strong>equation</strong></td><td>内置 PDE 方程组：N-S、泊松、拉普拉斯、线弹性、Volterra 等，支持自定义</td></tr>
<tr><td><strong>experimental</strong></td><td>实验性功能：贝塞尔曲线数据转换</td></tr>
<tr><td><strong>geometry</strong></td><td>几何构建：Mesh（STL）、点云、采样器、TimeDomain</td></tr>
<tr><td><strong>loss</strong></td><td>损失函数：MSE、积分损失、多任务聚合损失</td></tr>
<tr><td><strong>metric</strong></td><td>评估指标（类似 loss 但不回传梯度）</td></tr>
<tr><td><strong>optimizer</strong></td><td>优化器 + 学习率调度器（Adam、SGD、L-BFGS 等）</td></tr>
<tr><td><strong>probability</strong></td><td>HMC 高效采样方法</td></tr>
<tr><td><strong>solver</strong></td><td>训练/评估/预测/可视化封装，支持混合精度</td></tr>
<tr><td><strong>utils</strong></td><td>工具函数：数据加载、配置、权重管理、计时器</td></tr>
<tr><td><strong>validate</strong></td><td>配置合法性校验</td></tr>
<tr><td><strong>visualizer</strong></td><td>VTK/Matplotlib 可视化</td></tr>
</table>
<h3>4.3 关键工作流示例</h3>
<p><strong>定义方程：</strong></p>
<p><img src="物理信息机器学习框架调研/Paddle_17.png" alt="NS方程流量计算示意"></p>
<pre><code>equation = {
    "NavierStokes": ppsci.equation.NavierStokes(nu, rho, dim, False),
    "NormalDotVec": ppsci.equation.NormalDotVec(("u", "v", "w")),
}</code></pre>
<p><img src="物理信息机器学习框架调研/Paddle_18.png" alt="计算所用3D几何"></p>
<p><strong>定义内部 PDE 约束：</strong></p>
<pre><code>pde = ppsci.constraint.InteriorConstraint(
    equation["NavierStokes"].equations,
    {"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
    geom["interior_geo"],
    {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.pde},
    ppsci.loss.MSELoss("sum"),
    name="interior",
)</code></pre>
<p><strong>定义 3D STL 几何 + 边界约束：</strong></p>
<p><img src="物理信息机器学习框架调研/Paddle_20.png" alt="3D几何SDF采样"></p>
<pre><code># 加载 STL 几何
inlet_geo = ppsci.geometry.Mesh(cfg.INLET_STL_PATH)
# 归一化
inlet_geo = inlet_geo.translate(-np.array(cfg.CENTER)).scale(cfg.SCALE)

# 边界约束（入口抛物线速度分布）
bc_inlet = ppsci.constraint.BoundaryConstraint(
    {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
    {"u": inlet_u_ref_func, "v": inlet_v_ref_func, "w": inlet_w_ref_func},
    geom["inlet_geo"],
    {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_inlet},
    ppsci.loss.MSELoss("sum"),
    name="inlet",
)</code></pre>
<p><strong>Solver 封装：</strong></p>
<pre><code>solver = ppsci.solver.Solver(
    model, constraint, cfg.output_dir,
    optimizer, lr_scheduler,
    cfg.TRAIN.epochs, cfg.TRAIN.iters_per_epoch,
    eval_during_train=True,
    seed=cfg.seed,
    equation=equation, geom=geom,
    validator=validator, visualizer=visualizer,
)
solver.train()
solver.eval()
solver.visualize()</code></pre>
<blockquote><p>✅ <strong>核实</strong>：以上代码为 PaddleScience 标准 API 用法，架构设计参考了工业 CFD 工作流（前处理→求解→后处理），接口设计合理。</p></blockquote>
<h3>4.4 3D 几何依赖</h3>
<table>
<tr><th>依赖库</th><th>功能</th></tr>
<tr><td><strong>open3d</strong></td><td>STL 读取、点云转换与可视化</td></tr>
<tr><td><strong>pyvista</strong></td><td>VTK 封装，STL 处理、云图/色标可视化、导出 HTML</td></tr>
<tr><td><strong>pysdf</strong></td><td>计算点到面最小距离，支持批量计算（SDF 采样）</td></tr>
<tr><td><strong>pymesh</strong></td><td>网格生成与处理（赛桨基于 pymesh 统一管理其他依赖）</td></tr>
<tr><td><strong>pybind11</strong></td><td>Python 调用 C++ 模块（底层加速）</td></tr>
</table>
<blockquote><p>✅ <strong>核实</strong>：依赖库功能描述准确。pymesh 安装较困难，赛桨的 3D 功能对其依赖较重。</p></blockquote>
<p><strong>PINN 求解结果可视化：</strong></p>
<p><img src="物理信息机器学习框架调研/Paddle_21.png" alt="PINN求解结果可视化"></p>
<p><strong>3D 几何处理依赖库可视化：</strong></p>
<table>
<tr><th>依赖库</th><th>功能</th><th>示例</th></tr>
<tr><td>open3d</td><td>STL 读取、点云转换与可视化</td><td><img src="物理信息机器学习框架调研/Paddle_23.png" alt="open3d"></td></tr>
<tr><td>pyvista</td><td>VTK 封装、STL 处理、云图可视化、导出 HTML</td><td><img src="物理信息机器学习框架调研/Paddle_24.png" alt="pyvista"></td></tr>
<tr><td>pysdf</td><td>点到面 SDF 距离计算与采样</td><td><img src="物理信息机器学习框架调研/Paddle_25.png" alt="pysdf"></td></tr>
</table>
<hr>
<h2>五、降维与降阶方法：DMD / POD / DEIM / NARX</h2>
<blockquote><p>飞书原文中此部分内容标注为「AI 生成，不能完全保障真实」。以下核实并精炼。</p></blockquote>
<h3>5.1 核心区别</h3>
<table>
<tr><th>方法</th><th>本质</th><th>输入</th><th>输出</th><th>典型应用</th></tr>
<tr><td><strong>POD</strong></td><td>纯空间统计降维</td><td>全场快照数据</td><td>能量最优空间模态（无时间信息）</td><td>压缩、ROM 基函数构建</td></tr>
<tr><td><strong>DMD</strong></td><td>时空耦合动力学分解</td><td>时序全场数据</td><td>空间模态 + 单频 + 增长/衰减率</td><td>流场稳定性分析、短期预测</td></tr>
<tr><td><strong>DEIM</strong></td><td>非线性项插值加速</td><td>非线性项全场数据</td><td>少数关键插值点</td><td>配合 POD/DMD 加速 ROM 计算</td></tr>
<tr><td><strong>NARX</strong></td><td>时序神经网络预测</td><td>历史输出 + 外部输入</td><td>下一时刻预测值</td><td>工业过程单点时序预报</td></tr>
</table>
<h3>5.2 POD vs DMD 直观对比</h3>
<p><strong>POD</strong>（本征正交分解）：</p>
<ul>
<li>类似 PCA：按能量大小排序，少量模态保留绝大部分能量</li>
<li>模态正交，数学性质稳定</li>
<li><strong>局限</strong>：无时间动力学信息，单个模态含多频耦合，物理解释弱</li>
</ul>
<p><strong>DMD</strong>（动态模态分解）：</p>
<ul>
<li>假设系统近似线性动力学 x<sub>k+1</sub> = A·x<sub>k</sub></li>
<li>每个模态对应单一频率，自带增长/衰减系数</li>
<li><strong>优势</strong>：直观的物理可解释性（如判断涡脱落频率、流动稳定性）</li>
<li><strong>局限</strong>：强非线性系统精度下降</li>
</ul>
<blockquote><p>✅ <strong>核实</strong>：POD/DMD 的对比描述正确。POD 本质是 SVD 的空间模态截断，DMD 是对线性算子 A 的特征分解。两者经常组合使用。</p></blockquote>
<h3>5.3 DEIM 定位</h3>
<p>DEIM 不是独立建模算法，是 POD/DMD 降阶框架的配套加速工具。从海量网格中挑选少数关键插值点，用插值还原全场非线性项，将计算量压缩 10~100 倍。</p>
<p><strong>标准流程</strong>：POD/DMD 状态降维 → DEIM 加速非线性项求解</p>
<blockquote><p>✅ <strong>核实</strong>：DEIM 描述正确。它是非线性模型降阶（MOR）中的关键技术，在 CAE/CFD 数字孪生中有重要应用。</p></blockquote>
<h3>5.4 NARX 定位</h3>
<p>非线性自回归外生神经网络：将「历史输出（AR）+ 历史外部输入（X）」送入神经网络进行非线性预测。适用于工业传感器时序预测（温度、压力、振动等）。</p>
<blockquote><p>✅ <strong>核实</strong>：NARX 是成熟的工业时序预测方法，但只能处理单点时序，不能处理全场云图数据。与前述 POD/DMD/DEIM 属于完全不同的技术路线。</p></blockquote>
<h3>5.5 工程应用建议</h3>
<ul>
<li><strong>全场云图仿真</strong>：有限元生成全量数据 → DMD 提取动态模态 → DEIM 加速非线性项 → 快速仿真模型</li>
<li><strong>传感器时序预测</strong>：NARX 黑箱模型</li>
<li><strong>复杂多物理场数字孪生</strong>：POD/DMD 降阶 + DEIM 加速 + 神经网络混合模型</li>
</ul>
<hr>
<h2>六、NVIDIA PhysicsNeMo</h2>
<h3>6.1 概述</h3>
<p>NVIDIA PhysicsNeMo（前身 Modulus）是 NVIDIA 开源的大规模物理 AI 框架，基于 PyTorch，Apache 2.0 协议。当前正在升级至 v2.0。</p>
<p><strong>定位</strong>：为工业级数字孪生应用提供端到端的物理 AI 模型开发、训练和部署工具链。</p>
<h3>6.2 核心组件</h3>
<table>
<tr><th>组件</th><th>功能</th></tr>
<tr><td><strong>physicsnemo.models</strong></td><td>模型库：FNO、DeepONet、DoMINO、MeshGraphNet、XAeroNet、扩散模型、Transformer 等</td></tr>
<tr><td><strong>physicsnemo.datapipes</strong></td><td>数据管道：点云、mesh、工程/科学数据结构优化</td></tr>
<tr><td><strong>physicsnemo.distributed</strong></td><td>分布式训练：torch.distributed 封装，支持 FSDP、模型并行、多节点</td></tr>
<tr><td><strong>physicsnemo.sym</strong></td><td>符号 PDE：通过 SymPy 定义方程，自动计算物理信息损失（PINN）</td></tr>
<tr><td><strong>physicsnemo.curator</strong></td><td>数据管理：工程数据集整理与加速</td></tr>
</table>
<h3>6.3 关键特性</h3>
<ul>
<li><strong>GPU 优化</strong>：原生针对 NVIDIA GPU 优化，支持从单 GPU 到多节点集群</li>
<li><strong>预训练模型</strong>：NGC 模型库提供可直接下载的预训练模型</li>
<li><strong>领域扩展包</strong>：PhysicsNeMo CFD（CFD 专用推理）、PhysicsNeMo Curator（数据管理）、Earth-2 Studio（气候/天气 AI 模型）</li>
<li><strong>企业级应用</strong>：已被 Ansys、SimScale、Luminary Cloud、台积电、富士康等采用</li>
</ul>
<h3>6.4 与 DeepXDE / PaddleScience 的差异</h3>
<table>
<tr><th>特性</th><th>DeepXDE</th><th>PaddleScience</th><th>PhysicsNeMo</th></tr>
<tr><td>学术/工业定位</td><td>学术为主</td><td>国产工业验证</td><td>工业级大规模部署</td></tr>
<tr><td>分布式训练</td><td>基础支持</td><td>单机为主</td><td>原生多 GPU/多节点</td></tr>
<tr><td>模型架构丰富度</td><td>PINN + DeepONet</td><td>PINN + 数据 + 算子</td><td>全谱系（PINN + 算子 + GNN + 扩散 + Transformer）</td></tr>
<tr><td>预训练模型</td><td>无</td><td>无</td><td>NGC 模型库</td></tr>
<tr><td>生态集成</td><td>独立框架</td><td>PaddlePaddle 生态</td><td>NVIDIA 全栈（CUDA/Omniverse/DGX）</td></tr>
<tr><td>安装复杂度</td><td>低</td><td>中（需 PaddlePaddle）</td><td>推荐 NGC Container</td></tr>
</table>
<blockquote><p>✅ <strong>核实</strong>：PhysicsNeMo 前身是 NVIDIA Modulus（2023 年开源），2025 年更名为 PhysicsNeMo。框架确实以 PyTorch 为基础，强调 GPU 加速和大规模部署。</p></blockquote>
<h3>6.5 典型应用场景</h3>
<ul>
<li>CFD 替代模型：训练 AI 替代传统 CFD 求解器，推理加速 500x+</li>
<li>气动外形优化：如 Luminary Cloud 的 SHIFT-WING 跨音速机翼设计</li>
<li>半导体热仿真：Ansys 集成用于芯片设计优化</li>
<li>工厂数字孪生：富士康、台积电用于产线规划与运维</li>
</ul>
<hr>
<h2>七、选型建议</h2>
<h3>按场景推荐</h3>
<table>
<tr><th>场景</th><th>推荐框架</th><th>理由</th></tr>
<tr><td>学术研究、PINN 探索</td><td>DeepXDE</td><td>轻量、多后端灵活、论文复现友好</td></tr>
<tr><td>国内部署、国产化要求</td><td>PaddleScience</td><td>百度生态、中文文档、3D 几何支持好</td></tr>
<tr><td>工业级数字孪生、大规模部署</td><td>PhysicsNeMo</td><td>GPU 加速、分布式、预训练模型</td></tr>
<tr><td>瞬态仿真降阶加速</td><td>POD/DMD + DEIM</td><td>经典 ROM 路线，可与上述框架结合</td></tr>
</table>
<h3>综合建议</h3>
<ol>
<li>短期 PINN 探索可使用 DeepXDE 快速上手</li>
<li>如需复杂 3D 几何 + PINN 求解，PaddleScience 的 Mesh 模块更成熟</li>
<li>长期工业级数字孪生产品化方向，PhysicsNeMo 的生态和性能优势明显</li>
<li>降维方法（POD/DMD/DEIM）是加速瞬态仿真的关键技术，值得深入研究并与 PINN 框架结合</li>
</ol>
<hr>
<h2>八、参考资料</h2>
<ul>
<li>DeepXDE: <a href="https://github.com/lululxvi/deepxde" target="_blank">github.com/lululxvi/deepxde</a></li>
<li>PaddleScience: <a href="https://github.com/PaddlePaddle/PaddleScience" target="_blank">github.com/PaddlePaddle/PaddleScience</a></li>
<li>PhysicsNeMo: <a href="https://github.com/NVIDIA/physicsnemo" target="_blank">github.com/NVIDIA/physicsnemo</a></li>
<li>PhysicsNeMo 文档: <a href="https://docs.nvidia.com/deeplearning/physicsnemo/" target="_blank">docs.nvidia.com/deeplearning/physicsnemo</a></li>
<li>飞书知识库原文: <a href="https://mcn9i508eump.feishu.cn/wiki/FYdpwcGBSiCdMckulGdceHJCnug" target="_blank">Feishu Wiki</a></li>
</ul>
<h2>附录：Wiki 原文补充图片</h2>
<p>以下为飞书知识库原文中的其余图片，按出现顺序排列：</p>
<p><strong>DeepXDE 页面：</strong></p>
<ul>
<li><code>DeepXDE_01.png</code> — 框架概览图</li>
<li><code>DeepXDE_02.png</code> — PyTorch 后端选择提示</li>
</ul>
<p><strong>PaddleScience 页面：</strong></p>
<ul>
<li><code>Paddle_01.png</code> — 赛桨结构及功能梳理</li>
<li><code>Paddle_02.png</code> — 赛桨改进方案</li>
<li><code>Paddle_03.png</code> — 2D 定常方腔流（补充）</li>
<li><code>Paddle_06.png</code> — 管道流动（补充）</li>
<li><code>Paddle_08.png</code> — 3D 血管（补充）</li>
<li><code>Paddle_10.png</code> — 瞬态方腔流（补充）</li>
<li><code>Paddle_12.png</code> — 2D 平板变形（补充）</li>
<li><code>Paddle_14.png</code> — 汽车控制臂变形（补充）</li>
<li><code>Paddle_19.png</code> — NS 方程流量计算示意（补充）</li>
<li><code>Paddle_22.png</code> — HMC 采样结果（补充）</li>
</ul>
<blockquote><p>注：所有图片已下载至同级目录 <code>物理信息机器学习框架调研/</code> 下。</p></blockquote>`
  },

  // ===== 3. AI对话-DMD与POD及DEIM详细介绍 =====
  {
    id: "dmd-pod-deim-details",
    title: "DMD 与 POD 及 DEIM 详细介绍",
    date: "2026-06-25",
    category: "技术笔记",
    excerpt: "AI 对话整理：DMD、POD、DEIM、NARX 四种降阶方法的完整对比、Ansys Twin Builder ROM 技术及流体电磁瞬态仿真答疑。",
    content: `<blockquote><p>2026 年 6 月 25 日 · 内容由 AI 生成，不能完全保障真实</p></blockquote>
<h1>一、DEIM vs DMD vs NARX — 通俗易懂的详细解读</h1>
<p>DEIM、DMD、NARX 通俗完整解读。三者分属三大完全不同赛道，核心用途天差地别：</p>
<ul>
<li><strong>DMD</strong>：动态模态分解 → 流体 / 仿真 / 时序提取空间流动模式、降阶建模（纯数据驱动线性动力学）</li>
<li><strong>DEIM</strong>：离散经验插值法 → 配套 POD/DMD 的插值加速工具，不是独立建模算法，是降维加速插件</li>
<li><strong>NARX</strong>：非线性自回归外生神经网络 → 工业时序预测神经网络，带输入变量、专门做非线性预报</li>
</ul>
<p>下面分开用人话拆解，附带适用场景、优缺点、直观例子。</p>
<h2>一、DMD 动态模态分解 Dynamic Mode Decomposition</h2>
<h3>最简单一句话</h3>
<p>给一堆随时间变化的空间数据（比如流体速度场、应力云图、温度场视频帧），自动拆成固定空间形态 + 各自振荡频率，用少量模态还原整个动态过程，实现仿真降阶。</p>
<h3>通俗原理</h3>
<p>想象你拍了一段风吹树叶的视频（每一帧是二维空间数据）：把每一帧图像摊平成一列向量，拼成大矩阵；DMD 自动分解出几组基础画面（动态模态）：</p>
<ul>
<li>模态 1：树干不动的静态背景（频率≈0）</li>
<li>模态 2：大叶缓慢左右晃（低频）</li>
<li>模态 3：小叶快速抖动（高频）</li>
</ul>
<p>每个模态自带增长 / 衰减系数，能预测未来几帧树叶怎么动。数学本质：假设系统近似线性动力学 x<sub>k+1</sub>=Ax<sub>k</sub>，用 SVD 近似求解矩阵 A，分离空间结构与时频特征。</p>
<h3>能干什么</h3>
<ul>
<li>CFD 流体、CAE 多物理场仿真降阶（大幅替代昂贵有限元计算）</li>
<li>提取振动、流动、应力的关键振荡特征</li>
<li>短期时序场预测（全场云图预报）</li>
<li>故障特征提取（振动模态突变 = 设备异常）</li>
</ul>
<h3>优缺点</h3>
<ul>
<li>✅ 优点：纯数据、无网格方程、物理可解释（模态对应真实振荡）、计算快</li>
<li>❌ 缺点：强依赖线性假设，强非线性系统精度暴跌；无法直接引入外部输入（无控制变量）</li>
</ul>
<h2>二、DEIM 离散经验插值法 Discrete Empirical Interpolation Method</h2>
<h3>一句话定位</h3>
<p>DEIM不是独立建模算法，是 POD/DMD 降阶框架的配套加速工具，专门解决 "降维后非线性项计算依然很慢" 的痛点。</p>
<h3>通俗原理</h3>
<p>先用 POD/DMD 把全场数据压缩成少量模态（低维系数），但原模型里存在非线性项（湍流项、材料非线性、摩擦），每次计算仍要遍历全部网格点，速度上不去。DEIM 操作：</p>
<ul>
<li>从海量网格里，自动挑选少数关键插值点；</li>
<li>只用这几十个关键点的数值，插值还原全场非线性项；</li>
</ul>
<p>把上百万网格计算量压缩到几十点，加速 10~100 倍。类比：画人像不用铺满像素，只捕捉五官关键点位插值还原整张脸。</p>
<h3>和 DMD 的关系</h3>
<p>标准流程：POD/DMD 做状态降维 + DEIM 加速非线性项求解，二者经常成对出现，DEIM 不能单独用来建模预测。</p>
<h3>优缺点</h3>
<ul>
<li>✅ 优点：极大加速 CAE / 流体降阶模型在线计算；保证插值精度</li>
<li>❌ 缺点：依附 POD/DMD 存在；关键插值点固定，工况偏移后精度下降</li>
</ul>
<h2>三、NARX 非线性自回归外生神经网络</h2>
<h3>一句话通俗定义</h3>
<p>专门处理带外部输入的非线性时间序列预测神经网络，工业最常用的预报模型，既能看历史输出，又能利用外部控制参数。</p>
<p><strong>AR 部分（自回归）</strong>：看自己过去——预测明天温度，先看今天、昨天、前天的温度历史值；<strong>X 外生输入（Exogenous）</strong>：外部影响变量——车间温度还受：加热功率、室外气温、阀门开度这些外部输入影响。NARX 把「历史输出 + 历史外部输入」一起喂进神经网络，拟合非线性映射，输出下一时刻预测值。</p>
<h3>直观例子</h3>
<p>水泵振动预测：自回归项 AR：前 5 秒振动幅值（历史输出）；外生输入 X：转速、流量、压力（外部可控参数）。NARX 学习三者非线性关系，提前预判振动超标。</p>
<h3>优缺点</h3>
<ul>
<li>✅ 优点：天然适配非线性；支持多外部输入；工程落地简单、拟合能力强</li>
<li>❌ 缺点：纯黑箱、无物理解释；外推长期预测容易漂移；需要大量时序数据；无空间场建模能力（只能单点时序，不能处理云图全场数据）</li>
</ul>
<h2>四、三者核心对比（一眼分清）</h2>
<table>
<tr><th>维度</th><th>DMD</th><th>DEIM</th><th>NARX</th></tr>
<tr><td><strong>类别</strong></td><td>数据驱动线性动力学分解</td><td>非线性项插值加速工具</td><td>时序预测神经网络</td></tr>
<tr><td><strong>输入</strong></td><td>时序全场快照矩阵</td><td>非线性项全场快照</td><td>历史输出 + 外部输入</td></tr>
<tr><td><strong>输出</strong></td><td>空间模态 + 频率 + 增长率</td><td>关键插值点 + 插值矩阵</td><td>下一时刻预测值</td></tr>
<tr><td><strong>能否独立建模</strong></td><td>✅ 可独立做动力学分析</td><td>❌ 必须配套 POD/DMD</td><td>✅ 可独立做时序预测</td></tr>
<tr><td><strong>物理解释性</strong></td><td>✅ 强（模态 = 真实振荡）</td><td>⚠️ 中等（选点是数学操作）</td><td>❌ 无（纯黑箱）</td></tr>
<tr><td><strong>场建模能力</strong></td><td>✅ 全场云图</td><td>⚠️ 仅加速非线性项</td><td>❌ 只有单点时序</td></tr>
</table>
<h2>五、工程场景快速选择指南</h2>
<ul>
<li>你有仿真全场云图（温度、应力、流体速度），想提取流动特征、替代有限元仿真 → <strong>DMD</strong>；计算太慢搭配 <strong>DEIM</strong> 加速</li>
<li>做 POD/DMD 降阶后，非线性项计算耗时极高，需要提速 → <strong>DEIM</strong>（配套使用）</li>
<li>只有单点传感器时序（振动、压力），存在阀门、转速等外部控制量，需要做趋势预测 → <strong>NARX</strong></li>
</ul>
<h3>补充联动场景（工业 CAE 常见组合）</h3>
<p>大型多物理场数字孪生标准链路：有限元仿真生成全场数据 → DMD 提取动态模态构建降阶模型 → DEIM 加速非线性求解 → 得到快速仿真模型；如果仿真难以建模、只有传感器实测时序数据，则直接用 NARX 搭建黑箱预测模型。</p>
<hr>
<h1>二、DMD 和 POD 的区别在哪？DEIM 详细介绍</h1>
<h2>一、POD 与 DMD 完整通俗对比（核心区别一次性讲透）</h2>
<h3>基础定义一句话</h3>
<ul>
<li><strong>POD</strong>（本征正交分解）：纯空间统计降维，只看所有快照的能量分布，不关心时间演化关系；目标是用最少模态还原全场能量，无动力学、无频率、无增长衰减。</li>
<li><strong>DMD</strong>（动态模态分解）：时空耦合动力学分解，强制捕捉时序递推关系 x<sub>t+1</sub>=Ax<sub>t</sub>；每个模态自带单一频率 + 增长 / 衰减系数，能描述流动随时间怎么变、会不会发散失稳。</li>
</ul>
<h3>POD 视角（风吹树叶）</h3>
<p>把所有帧堆在一起统计：模态 1：树干（占总能量 70%）；模态 2：大片树叶同步晃动（能量 20%）；模态 3：细碎小叶抖动（能量 5%）。只告诉你哪块区域动得最猛，但分不清这片晃动是低频慢摇还是高频抖，也不知道接下来会越晃越大还是慢慢平息。单个 POD 模态里会混合多个频率，物理上很难单独解释振荡机理。</p>
<h3>DMD 视角</h3>
<p>按时间顺序看画面变化，拆出独立振荡单元：模态 1：静态树干（频率 0，恒定不变）；模态 2：大叶低频缓慢摆动（衰减，越晃越弱）；模态 3：小叶高频快速抖动（微弱增长，会越来越剧烈）。每一组空间形态只对应一个固定频率，同时标注该结构是衰减、稳定还是发散，直接判断流动稳定性、振动风险。</p>
<h3>六大核心差异</h3>
<table>
<tr><th>对比维度</th><th>POD</th><th>DMD</th></tr>
<tr><td><strong>核心目标</strong></td><td>空间能量压缩（降维）</td><td>时空动力学分解（辨识+预测）</td></tr>
<tr><td><strong>数学工具</strong></td><td>SVD 奇异值分解</td><td>SVD + 特征分解</td></tr>
<tr><td><strong>输出内容</strong></td><td>能量排序空间模态（无频率）</td><td>空间模态 + 单频 + 增长率</td></tr>
<tr><td><strong>时间信息</strong></td><td>❌ 无</td><td>✅ 完整（频率、衰减/发散）</td></tr>
<tr><td><strong>模态正交性</strong></td><td>✅ 严格正交</td><td>⚠️ 不保证正交</td></tr>
<tr><td><strong>重构压缩效率</strong></td><td>✅ 最优（能量意义上）</td><td>⚠️ 不如 POD</td></tr>
</table>
<h3>优缺点总结</h3>
<p><strong>POD 优势：</strong>压缩效率极高，少量模态就能保留绝大部分能量；模态正交，数学性质稳定，适合做 Galerkin 投影构建降阶模型；计算简单、鲁棒，噪声容忍度高。</p>
<p><strong>POD 短板：</strong>无时间动力学信息，不能提取频率、无法判断流动是否发散；模态多频耦合，物理解释弱。</p>
<p><strong>DMD 优势：</strong>时空绑定，模态物理意义清晰（对应单一振荡结构）；输出频率、增长率，直接用于稳定性分析、故障预警；原生支持短期全场预测。</p>
<p><strong>DMD 短板：</strong>强假设系统近似线性，强非线性工况精度暴跌；模态不保证正交，重构压缩效率普遍不如 POD；对时序采样间隔、噪声更敏感。</p>
<h3>工程怎么选</h3>
<ul>
<li>只想压缩海量网格快照、做仿真加速基底 → <strong>POD</strong></li>
<li>需要分析振动 / 流动振荡频率、判断失稳、预测未来流场 → <strong>DMD</strong></li>
<li>完整数字孪生降阶：<strong>POD</strong> 做基底 + DEIM 加速非线性；<strong>DMD</strong> 单独做动力学特征提取</li>
</ul>
<h2>二、DEIM 离散经验插值法 超详细完整介绍</h2>
<h3>定位：不是独立建模算法，是 POD-Galerkin 降阶的「超加速插件」</h3>
<h3>核心痛点（为什么需要 DEIM）</h3>
<p>标准 POD-Galerkin 降阶只优化线性项，但 CAE/CFD 里大量非线性项（湍流、塑性、摩擦、化学反应、热辐射）有致命缺陷：哪怕你把状态压缩到几十阶模态，每一步迭代计算非线性项时，依然要遍历原始几十万网格点，计算量和全阶仿真几乎没区别，降阶完全失去速度优势。DEIM 就是专门解决「非线性项计算成本居高不下」的算法。</p>
<h3>通俗原理（抓关键点插值还原全场）</h3>
<p>类比：一幅百万像素应力云图，不用遍历全部像素算应力非线性；DEIM 自动挑几十张最敏感、最具代表性的网格测点，只在这几十个点精确计算非线性值，其余上百万网格全部通过线性插值还原。</p>
<h3>完整三步标准流程</h3>
<ol>
<li><strong>步骤 1：采集非线性快照，构建非线性基底</strong>：把每个时间步的全场非线性项 F(u(t)) 作为一列拼成矩阵，对该矩阵做 POD-SVD，提取前 m 阶正交基底 Z（m 远小于原始网格数 N，一般取 20~100）。F(u)≈Z·c，c 是低维插值系数向量。</li>
<li><strong>步骤 2：贪心算法挑选 DEIM 插值点（核心）</strong>：取基底第一列，找绝对值最大的网格行号，记为第 1 个插值点；用已选点构造插值矩阵，求残差，在残差最大位置选下一个点；循环直到选满 m 个插值点，保存索引矩阵 P（仅保留选中网格的行）。作用：保证这 m 个点能唯一、稳定插值还原整个非线性场。</li>
<li><strong>步骤 3：插值近似，把计算量从 N 降到 m</strong>：原始非线性计算复杂度：O(N)（N = 网格总数，几十万）。DEIM 近似后：只在 m 个选中点计算 P·F(u)，再插值还原全场：F<sub>DEIM</sub>(u)=Z·(PZ)<sup>−1</sup>·PF(u)。复杂度直接变为 O(m)，m 通常几十，速度提升几十～上百倍。</li>
</ol>
<h3>DEIM 完整优缺点</h3>
<p><strong>优点：</strong></p>
<ul>
<li>极致加速：非线性项计算量从百万网格压缩到几十测点，在线仿真、参数扫描、数字孪生实时推演成为可能</li>
<li>精度可控：插值误差有理论上界，可通过增加插值点数量平衡速度与精度</li>
<li>完全配套 POD-Galerkin，线性代数实现，无训练、无神经网络、无调参</li>
<li>通用性强：流体、热传导、结构塑性、电磁、电池多物理场全部适用</li>
</ul>
<p><strong>缺点：</strong></p>
<ul>
<li>不能单独使用：必须依赖 POD 生成非线性基底，脱离 POD 无法工作</li>
<li>插值点固定：训练快照覆盖不到的极端工况，插值误差会急剧增大</li>
<li>强非线性突变场景（冲击波、剧烈失稳）需要更多插值点，加速效果下降</li>
<li>离线成本：提前选点、构造插值矩阵需要一次全阶仿真快照，离线预处理有少量开销</li>
</ul>
<h3>标准工程组合链路（POD+DEIM）</h3>
<ol>
<li>全阶有限元 / CFD 仿真，输出各时刻全场状态快照、全场非线性项快照</li>
<li>对状态快照做 POD，得到状态低维基，构建 Galerkin 降阶方程组</li>
<li>对非线性项快照单独 POD，用 DEIM 贪心算法选出关键网格插值点</li>
<li>将 DEIM 插值算子代入降阶方程，替换原始全场非线性计算</li>
<li>得到 POD-DEIM 降阶模型，离线预处理完成，在线实时推演速度提升几十倍</li>
</ol>
<h3>拓展变种（工程常用）</h3>
<ul>
<li><strong>W-DEIM</strong> 加权 DEIM：对高应力、高流速区域加大权重，优先捕捉关键区域插值点</li>
<li><strong>自适应 DEIM</strong>：动态增减插值点，工况偏移时自动补充测点</li>
<li><strong>分区 DEIM</strong>：多物理场分开选插值点，适配耦合仿真</li>
</ul>
<h3>极简总结</h3>
<p>POD = 给全场数据做空间压缩；DMD = 给时序数据做动力学 + 频率分解；DEIM = 给 POD 降阶里的非线性项做「智能采样加速工具」，三者经常联合用于 CAE 数字孪生模型降阶。</p>
<hr>
<h1>三、Ansys Twin Builder ROM：稳态和瞬态降阶方法</h1>
<h2>总纲区分</h2>
<ul>
<li><strong>Static ROM</strong>（稳态 / 静力学 ROM）：无时间导数、只处理定常场（稳态温度、静应力、稳态流场、静磁场），核心：POD (SVD) 模态压缩 + 元模型插值，纯代数映射，不含动力学</li>
<li><strong>Dynamic ROM</strong>（瞬态 / 动态 ROM）：带时间演化，分线性 LTI/LPV、非线性 POD-Galerkin+DEIM、数据辨识 DMD三类，专门处理随时间变化的瞬态场（冲击、振动、瞬态热、瞬态流体）</li>
</ul>
<p>DMD 仅作为瞬态特征提取手段，不单独作为 Twin Builder 标准 ROM 求解基底；DEIM 只配套非线性瞬态 POD-Galerkin 加速非线性项。</p>
<h2>一、稳态 Static ROM 完整降阶方法</h2>
<h3>底层核心算法：POD（SVD 奇异值分解）</h3>
<p>稳态多参数 DOE 采样（拉丁超立方 LHS 最优采样），每组参数对应一张全场快照（应力 / 温度 / 流速云图），全部快照拼成矩阵做 SVD 分解，提取能量主导正交模态基底，把百万网格高维场压缩到几十阶模态系数。公式：u(p)≈Φ·a(p)</p>
<ul>
<li>Φ：POD 正交模态基底（SVD 左奇异向量）</li>
<li>p：输入参数（载荷、功率、流速、几何尺寸）</li>
<li>a(p)：低维模态系数（仅几十维）</li>
</ul>
<h3>Twin Builder 内置逻辑</h3>
<p>稳态没有时间序列，只有参数 - 场映射：Workbench/Mechanical/Fluent 批量 DOE 扫参，输出多组稳态全场快照；Static ROM Builder 对全部快照做 SVD/POD，自动截断低能量模态（保留 99% 总能量）；对每一个模态系数 a(p) 单独训练元模型（插值代理模型），实现任意新参数下快速预测模态系数；模态基底 × 预测系数 = 实时重构全场云图。</p>
<h3>模态系数插值（元模型）</h3>
<p>Twin Builder 内置 4 种插值算法，自动择优：</p>
<ul>
<li><strong>响应面 RSM（多项式）</strong>：线性 / 二次多项式，弱非线性、少参数场景</li>
<li><strong>Kriging 克里金</strong>：全局最优插值，中等非线性，工程最常用</li>
<li><strong>SVR 支持向量回归</strong>：强非线性、参数耦合强</li>
<li><strong>GARS 聚合元模型</strong>：多模型加权融合，高精度优化场景</li>
</ul>
<h3>Static ROM 有没有 DEIM？</h3>
<p>稳态 ROM 原生不使用 DEIM。原因：稳态方程是纯代数方程，无迭代非线性项遍历，一次投影即可完成全场重构，不存在 "每步计算都要遍历全网格" 的痛点，DEIM 仅用于瞬态非线性迭代加速。</p>
<h2>二、瞬态 Dynamic ROM 三大主流降阶路线</h2>
<h3>路线 1：LTI 线性时不变 ROM（纯线性瞬态，无参数变化）</h3>
<p><strong>适用场景</strong>：纯线性热传导、无湍流恒定流场散热、线性结构振动、无饱和电磁系统；系统参数固定，仅激励随时间变化。</p>
<p><strong>底层降阶算法</strong>：</p>
<ul>
<li><strong>平衡截断 Balanced Truncation</strong>（状态空间首选）：将有限元离散偏微分方程转为一阶线性状态空间 x˙=Ax+Bu, y=Cx+Du，通过平衡变换分离可控/可观状态，截断弱贡献状态，得到几十阶低阶线性模型</li>
<li><strong>Krylov Arnoldi 降阶</strong>：频域传递函数降阶，适合电磁、热系统频域响应拟合</li>
<li><strong>辅助：POD 模态叠加</strong>（结构振动谐响应）</li>
</ul>
<h3>路线 2：LPV 线性变参数 ROM（分段线性、弱非线性）</h3>
<p><strong>适用场景</strong>：冷却流量可变电池热、转速变化电机、载荷缓慢变化结构；每个固定参数下系统线性，参数连续变化导致整体非线性。</p>
<p><strong>原理</strong>：多组固定参数下分别构建 LTI 线性降阶模型；对降阶后的 A/B/C/D 状态矩阵做参数插值（Kriging）；任意中间参数实时插值得到对应低阶线性系统，实现变参数瞬态仿真。</p>
<h3>路线 3：非线性瞬态 ROM（POD-Galerkin + DEIM）</h3>
<p>核心组合：POD 基底投影 + DEIM 离散经验插值（Twin Builder 非线性瞬态标准方案）。</p>
<p><strong>POD-Galerkin 投影降阶（基础）</strong>：瞬态偏微分方程 Mu˙+Ku+F<sub>nl</sub>(u)=Bu<sub>in</sub>，F<sub>nl</sub>(u) 为全场非线性项（湍流、对流、塑性应力）。用 POD 基底Φ做 Galerkin 正交投影，把百万自由度方程压缩为 r 阶低维方程。致命问题：非线性项 Φ<sup>T</sup>F<sub>nl</sub>(Φa) 每次迭代仍要遍历全部网格 N。</p>
<p><strong>DEIM 离散经验插值法</strong>（Twin Builder 非线性瞬态强制配套加速）：完全解决非线性项网格遍历开销，是瞬态非线性 ROM 必不可少组件。</p>
<ul>
<li>提取所有时间步非线性项快照，做二次 POD 得到非线性基底Z</li>
<li>贪心算法自动挑选 m 个关键网格插值点（m=20~100，远小于网格总数 N）</li>
<li>构造插值算子，仅在 m 个关键点计算非线性值，插值还原全场非线性场</li>
<li>代入 Galerkin 方程，非线性项计算复杂度从 O(N)→O(m)，提速 50~1000 倍</li>
</ul>
<h3>路线 4：数据驱动黑箱瞬态 ROM（无物理投影，纯辨识）</h3>
<p>内置算法：NARX 神经网络、LSTM、高斯过程 GPR、向量拟合 Vector Fitting（频域电磁/热系统）。</p>
<p><strong>适用场景</strong>：只有传感器测点时序、无完整网格场数据，无法做 POD 投影，纯输入输出黑箱建模；缺点：无全场云图重构能力，仅输出单点时序。</p>
<h2>三、稳态 Static ROM vs 瞬态 Dynamic ROM 核心算法对比</h2>
<table>
<tr><th>对比维度</th><th>Static ROM（稳态）</th><th>Dynamic ROM（瞬态）</th></tr>
<tr><td><strong>时间演化</strong></td><td>❌ 无</td><td>✅ 有</td></tr>
<tr><td><strong>核心算法</strong></td><td>POD + Kriging 插值</td><td>POD-Galerkin / LTI / LPV / NARX</td></tr>
<tr><td><strong>是否求解 ODE</strong></td><td>❌ 纯代数映射</td><td>✅ 线性 ODE / 非线性 ODE</td></tr>
<tr><td><strong>是否用 DEIM</strong></td><td>❌ 不需要</td><td>✅ 非线性瞬态标配</td></tr>
<tr><td><strong>在线计算量</strong></td><td>O(r) 代数运算</td><td>O(r)~O(m) ODE 积分</td></tr>
<tr><td><strong>典型应用</strong></td><td>静应力、稳态热、定常流场</td><td>冲击、振动、瞬态热、瞬态CFD</td></tr>
</table>
<h2>四、工程选型速查（Twin Builder 实际使用）</h2>
<ul>
<li><strong>稳态工况</strong>（静力学、稳态热、稳态流场）→ Static ROM：POD (SVD)+Kriging 响应面，无 DEIM</li>
<li><strong>线性瞬态</strong>（无湍流、无塑性、无变参数）→ LTI ROM：平衡截断 / Arnoldi，纯线性状态空间</li>
<li><strong>变参数线性瞬态</strong>（流量 / 转速缓慢变化）→ LPV ROM：多组 LTI 插值</li>
<li><strong>强非线性瞬态</strong>（CFD 湍流、塑性、辐射、自然对流）→ 非线性 Dynamic ROM：POD-Galerkin + DEIM（行业标准组合）</li>
<li><strong>只有测点时序、无全场网格数据</strong> → 数据驱动 ROM：NARX、向量拟合、LSTM 黑箱辨识</li>
<li><strong>只分析振荡 / 失稳机理，不做快速仿真</strong> → 单独 DMD 分解提取模态频率、增长率，不生成可计算 ROM</li>
</ul>
<hr>
<h1>四、流体、电磁场景：瞬态 ROM 是方程计算还是数据拟合？</h1>
<h2>核心结论先讲透</h2>
<p>Twin Builder 里标准瞬态投影 ROM（流体、电磁主流）：<strong>是方程求解，不是纯数据拟合</strong>。</p>
<p>流体（Fluent 瞬态 CFD）、电磁（Maxwell 瞬态电磁场）、瞬态热、结构振动，软件默认用投影式 ROM（POD-Galerkin / LTI 平衡截断 / LPV）：底层依托原始物理控制方程（NS 流体方程、麦克斯韦电磁方程、热传导波动方程）；全阶仿真快照只用来提取 POD 模态基底 / 降阶算子矩阵，不直接拟合输入输出时序；在线运行时数值求解低阶常微分方程组 ODE，每一步按物理方程迭代计算，不是插值 / 神经网络查表预测。</p>
<p>只有一类瞬态 ROM 是纯数据驱动拟合：黑箱辨识 ROM（NARX/LSTM/向量拟合），无方程、纯时序拟合，仅当你只有传感器测点、没有全场网格数据时才用。</p>
<h2>一、分场景拆解：投影式 ROM（方程驱动）</h2>
<h3>（一）线性电磁 / 线性热 / 无湍流流体：LTI / LPV ROM</h3>
<p><strong>电磁例子（Maxwell 瞬态）</strong>：原始全阶离散方程 Mu˙+Ku=B·i(t)，u：全域磁势/电场（百万网格自由度），i(t)：线圈激励电流。离线用全阶瞬态仿真快照，通过平衡截断/Arnoldi Krylov 把百万阶矩阵投影压缩成几十阶低阶矩阵，得到低阶线性动力学微分方程（状态空间）。在线运行直接数值积分求解这个 ODE，按时间步迭代更新 a(t)，再用模态基底还原全场电磁场云图。</p>
<p><strong>关键特点</strong>：快照只用来构造降阶矩阵，不拟合输入输出曲线；每一步计算严格遵循电磁控制方程结构，满足守恒律；属于灰箱、方程驱动，不是纯数据插值。</p>
<h3>（二）非线性流体（带对流、湍流、相变）：POD-Galerkin + DEIM</h3>
<p>流体 NS 方程自带强非线性对流项，是最典型的方程投影 + DEIM 超缩减流程：原始全阶控制方程（有限体积离散）Mu˙+Ku+F<sub>nl</sub>(u)=B·u<sub>in</sub>(t)，F<sub>nl</sub> 为对流、湍流、压力耦合等全场非线性项。离线全阶仿真输出全场速度/压力快照、非线性项快照，做两次 POD：状态 POD 得到空间模态基底Φ，u≈Φa；非线性项 POD 生成 DEIM 插值基底，贪心选出几十关键网格点。Galerkin 投影把百万阶 PDE 压缩为低阶 ODE。</p>
<p><strong>重点区分：快照的作用≠拟合</strong>——快照仅用来提取模态基底、生成 DEIM 插值点、预计算投影矩阵；在线推演时没有任何插值 / 神经网络拟合步骤，是严格求解投影后的流体控制方程。</p>
<h2>二、两种瞬态 ROM 本质对比</h2>
<table>
<tr><th>维度</th><th>投影式 POD-Galerkin/LTI（方程计算型）</th><th>黑箱 NARX/LSTM（纯数据拟合型）</th></tr>
<tr><td><strong>底层机制</strong></td><td>数值求解低阶 ODE（投影后的 PDE）</td><td>神经网络前向推理（输入输出映射）</td></tr>
<tr><td><strong>是否用控制方程</strong></td><td>✅ 全程依赖 PDE/ODE</td><td>❌ 完全不用</td></tr>
<tr><td><strong>快照作用</strong></td><td>离线提取模态基底/投影矩阵</td><td>训练数据直接拟合</td></tr>
<tr><td><strong>全场云图输出</strong></td><td>✅ 模态重构全场</td><td>❌ 仅单点时序</td></tr>
<tr><td><strong>物理守恒</strong></td><td>✅ 满足（方程约束）</td><td>❌ 无保证</td></tr>
<tr><td><strong>外推能力</strong></td><td>✅ 较好（物理约束）</td><td>❌ 弱（离训练域即崩）</td></tr>
</table>
<h2>三、一句话通俗总结</h2>
<ul>
<li>用 Fluent/Maxwell 导出全场快照做的瞬态 ROM：<strong>先拿数据压缩方程，运行时解方程算结果</strong></li>
<li>只用传感器时间序列训练 NARX 模型：<strong>纯靠数据拟合映射，没有物理方程参与计算</strong></li>
<li>工业流体、电磁数字孪生实时仿真，全部走第一种「方程投影 + 低阶 ODE 求解」路线</li>
</ul>
<hr>
<h1>五、官方文档、论文、博客等依据</h1>
<h2>一、Ansys 官方顶层定义文档</h2>
<p><strong>What is a Reduced Order Model?</strong>（官方 ROM 分类标准）<br>
链接：<a href="https://www.ansys.com/simulation-topics/what-is-a-reduced-order-model" target="_blank">ansys.com</a></p>
<p>核心原文摘抄：ROM 分为 Intrusive（Model-Based，方程驱动侵入式）/ Non-Intrusive（Data-Based，纯数据驱动）。</p>
<ul>
<li><strong>Intrusive</strong>（流体/电磁瞬态 Dynamic ROM 主流）：直接访问高保真求解器内部 PDE 方程，通过 Galerkin 投影、平衡截断、Arnoldi Krylov 生成低阶微分方程/状态空间 ODE，在线数值求解方程，保留物理守恒律；POD-Galerkin、LTI、LPV、POD+DEIM 全部属于此类。</li>
<li><strong>Non-Intrusive</strong>（NARX/LSTM 黑箱）：仅使用仿真输出时序数据，不接触原始控制方程，输入输出映射纯拟合，无微分方程求解，无法重构全场云图。</li>
</ul>
<h2>二、Twin Builder 官方帮助手册</h2>
<p><strong>Twin Builder 官方帮助文档 Dynamic ROM 章节</strong>（2024R2/2025R1）<br>
手册明确划分两类 Dynamic ROM：</p>
<ul>
<li><strong>Projection-based Dynamic ROM</strong>（投影型，流体/电磁默认）：线性电磁/热用 Balanced Truncation/Arnoldi Krylov 生成 LTI 状态空间；非线性 CFD/塑性用 POD-Galerkin + DEIM 超缩减。在线仿真逻辑：时间步积分求解低阶 ODE，模态重构全场。</li>
<li><strong>Data-driven Identification ROM</strong>（辨识黑箱）：包含 NARX、向量拟合、GPR、LSTM，无任何微分方程求解，仅做输入输出时序插值预测，不能输出全场云图。</li>
</ul>
<h2>三、DEIM、POD-Galerkin 核心学术论文</h2>
<ul>
<li><strong>DEIM 原始论文</strong>：Chaturantabut & Sorensen (2008) — "Discrete Empirical Interpolation Method for nonlinear model reduction"</li>
<li><strong>CFD POD-Galerkin 工程实现</strong>：Practical implementation aspects of Galerkin ROMs based on POD for CFD (2020)</li>
<li><strong>时域麦克斯韦 POD-Galerkin</strong>：Parametric POD-Galerkin model order reduction for the time-domain Maxwell's equations (2019)</li>
</ul>
<h2>五、依据对应核心总结</h2>
<ol>
<li><strong>流体、电磁瞬态 Dynamic ROM 是否构建低阶动力学微分方程？</strong>官方文档 + 论文全部证实：是的。线性电磁/无湍流流体：LTI 低阶线性 ODE；湍流/塑性非线性流体：带 DEIM 加速的非线性低阶 ODE。</li>
<li><strong>瞬态投影 ROM 是数据拟合还是方程计算？</strong>官方明确二分法：POD-Galerkin/LTI/LPV（流体电磁主流）是方程计算为主；NARX/LSTM 等辨识 ROM 是纯数据驱动拟合。</li>
<li><strong>DEIM 定位</strong>：DEIM 是 POD-Galerkin 侵入式 ROM 的非线性加速工具，仅存在于非线性瞬态投影 ROM。</li>
<li><strong>稳态 Static ROM 区别</strong>：稳态无时间导数，仅 POD 模态 + Kriging/RSM 元模型插值，不构建动力学 ODE，无 DEIM。</li>
</ol>`
  },

  // ===== 4. Ansys ROM Guide =====
  {
    id: "ansys-rom-guide",
    title: "Ansys Twin Builder 降阶模型（ROM）技术指南",
    date: "2026-07-09",
    category: "调研",
    excerpt: "Ansys Twin Builder ROM 全流程详解：POD、DMD、DEIM 三大算法原理、Python 示例代码、工程选型速查。",
    content: `<h2>🔬 什么是降阶模型（ROM）</h2>
<p style="font-size:1.05em;line-height:1.8;margin-bottom:20px;">
降阶模型（<strong>R</strong>educed <strong>O</strong>rder <strong>M</strong>odel）的核心思想很简单：<strong>用一个低维数学模型替代高保真全阶仿真（FOM），在损失可忽略精度的情况下，将计算时间从小时级压缩到秒级甚至毫秒级。</strong>
</p>
<p>典型场景：一个 CFD 流场仿真可能有 <strong>10⁶–10⁸ 自由度</strong>，仿真一次需要数小时。ROM 将其压缩到 <strong>几十阶</strong>（10–100），在线推演仅需毫秒。这使实时控制、数字孪生、多查询优化（设计空间探索）成为可能。</p>

<h3>两大技术路线</h3>
<table>
<tr><th style="width:15%;">维度</th><th style="width:42%;">侵入式 ROM<br>Intrusive / Projection-based</th><th style="width:42%;">非侵入式 ROM<br>Non-intrusive / Data-driven</th></tr>
<tr><td><strong>核心思想</strong></td><td>将控制方程（PDE）投影到低维子空间，重新推导出低维 ODE 系统</td><td>纯数据驱动：从仿真快照学习输入→输出映射，<strong>不接触控制方程</strong></td></tr>
<tr><td><strong>是否需要方程</strong></td><td><span class="badge badge-red">必须</span> 需要访问 PDE 残差和算子</td><td><span class="badge badge-green">不需要</span> 只靠快照数据</td></tr>
<tr><td><strong>代表方法</strong></td><td>POD-Galerkin、平衡截断（Balanced Truncation）、Krylov 子空间</td><td>Kriging 响应面、神经网络（LSTM/NARX）、径向基函数插值</td></tr>
<tr><td><strong>精度</strong></td><td><span class="badge badge-green">高</span> 保留了物理结构（能量守恒、对称性等）</td><td><span class="badge badge-yellow">中–高</span> 依赖采样质量和数据量</td></tr>
<tr><td><strong>离线成本</strong></td><td>需要多次全阶仿真 + SVD 分解</td><td>需要大量仿真样本（DOE）</td></tr>
<tr><td><strong>在线速度</strong></td><td><span class="badge badge-green">极快</span> 毫秒级 ODE 积分</td><td><span class="badge badge-green">快</span> 代数插值或神经网络推理</td></tr>
<tr><td><strong>对非线性</strong></td><td><span class="badge badge-yellow">需 DEIM 辅助</span> 否则复杂度仍为 O(N)</td><td><span class="badge badge-green">天然处理</span> 无需额外措施</td></tr>
<tr><td><strong>外推能力</strong></td><td><span class="badge badge-green">较好</span> 物理方程约束了外插行为</td><td><span class="badge badge-red">较弱</span> 超出训练域精度急剧下降</td></tr>
</table>

<div class="tip"><strong>选型直觉：</strong>有控制方程 + 需要物理保真 → 侵入式；有数据但没方程 / 只关心单点输出 → 非侵入式。Twin Builder 两种都支持。</div>

<h2>🧮 三大核心算法</h2>
<p>POD、DMD、DEIM 是降阶模型领域的三大基石。它们解决不同的问题，实践中经常组合使用。</p>

<div class="algo-grid">
  <div class="algo-card pod">
    <h4>🔷 POD — 本征正交分解</h4>
    <p><strong>本质：SVD 空间压缩。</strong>将流场快照矩阵 <strong>X</strong>（每列一帧）做奇异值分解：<strong>X = U Σ Vᵀ</strong>。取前 r 列 U 构成模态基 Φ。流场近似为：<strong>u(x,t) ≈ Σᵢ φᵢ(x) · aᵢ(t)</strong>——空间模态×时间系数。</p>
    <p><strong>能量截断：</strong>保留 Σ 中前 r 个奇异值使 <strong>Σᵢ₌₁ʳ σᵢ² / Σᵢ σᵢ² ≥ 99%</strong>，百万自由度 → 几十阶。<br>
    <strong>Galerkin 投影：</strong>将 NS 方程投影到 POD 基上得到低维 ODE：<strong>ȧ = f(a)</strong>。<br>
    <strong>典型加速：650×</strong>（涡激振动，Stabile 2018）。</p>
  </div>

  <div class="algo-card dmd">
    <h4>💜 DMD — 动态模态分解</h4>
    <p><strong>本质：时空动力学分解。</strong>假设流场遵循线性动力学 <strong>xₖ₊₁ = A xₖ</strong>，从快照序列构造数据矩阵 <strong>X</strong> 和移位矩阵 <strong>X'</strong>。通过 SVD 近似求解算子 <strong>A</strong>，对其特征分解获得每个模态的<strong>单一频率 + 增长/衰减率</strong>。</p>
    <p><strong>输出：</strong>复特征值 λ（频率、增长/衰减率）+ 复特征向量 φ（空间模态）。<br>
    <strong>应用：</strong>流动稳定性分析、失稳预警、故障特征提取、POD 的时空替代方案。<br>
    <strong>局限：</strong>强线性假设——对非线性工况（湍流、激波）精度下降。</p>
  </div>

  <div class="algo-card deim">
    <h4>🟠 DEIM — 离散经验插值</h4>
    <p><strong>本质：非线性项加速插值。</strong>POD-Galerkin 降阶后，非线性项 <strong>N(u)</strong> 仍需在全网格（百万点）上计算→速度从毫秒退回秒级。DEIM 用贪心算法从百万网格中挑出几十个关键插值点，<strong>只在选点计算非线性值</strong>。</p>
    <p><strong>公式：</strong>F ≈ U<sub>f</sub> (PᵀU<sub>f</sub>)⁻¹ PᵀF——将全网格非线性值 PᵀF（仅选点）通过插值矩阵恢复全场。<br>
    <strong>加速：10–73×</strong>，选 50–90 个点误差 &lt;1%。<br>
    <strong>改进：</strong>Q-DEIM（Drmac &amp; Gugercin 2015）用列主元 QR 替代贪心选点。</p>
  </div>
</div>

<h3>六维对比表</h3>
<table>
<tr><th>维度</th><th>POD</th><th>DMD</th><th>DEIM</th></tr>
<tr><td><strong>核心数学</strong></td><td>SVD 奇异值分解</td><td>SVD + 特征分解</td><td>SVD + 贪心选点</td></tr>
<tr><td><strong>解决的问题</strong></td><td>空间降维（找最佳低维基）</td><td>时空动力学分析（频率/模态）</td><td>非线性项加速（避免全网格）</td></tr>
<tr><td><strong>输出</strong></td><td>空间模态 + 时间系数</td><td>模态频率、增长率、空间模式</td><td>插值点索引 + 插值矩阵</td></tr>
<tr><td><strong>是否需要方程</strong></td><td>否（纯数据压缩）</td><td>否（纯数据驱动）</td><td>是（需配合 Galerkin）</td></tr>
<tr><td><strong>典型加速</strong></td><td>100–1000×</td><td>N/A（分析工具，非加速器）</td><td>10–73×（叠加 POD 之上）</td></tr>
<tr><td><strong>组合使用</strong></td><td colspan="3"><strong>POD（空间压缩）→ Galerkin 投影（物理约束）→ DEIM（非线性加速）= 工业级非线性 ROM</strong></td></tr>
</table>

<h2>⚙️ Twin Builder ROM 全流程</h2>
<p>Ansys Twin Builder 提供两大类 ROM 构建路径：<strong>Static ROM</strong> 处理稳态问题，<strong>Dynamic ROM</strong> 处理瞬态问题。每类内部又根据物理特性选择不同算法。</p>

<div class="grid2">
  <div class="tier-box tier-cyan">
    <h3>📐 Static ROM（稳态）</h3>
    <p><strong>适用：</strong>静力学、稳态热传导、稳态流场等与时间无关的问题。<br>
    <strong>方法：</strong>POD（SVD）提取空间模态，配合<strong>响应面（Response Surface）</strong>或<strong>克里金插值（Kriging）</strong>拟合参数空间中的模态系数。<br>
    <strong>输入：</strong>设计参数（几何尺寸、材料属性、边界条件）→ <strong>输出：</strong>全场分布（位移、温度、压力）。<br>
    <strong>本质：</strong>非侵入式——只管输入到输出的映射，不碰控制方程。</p>
  </div>
  <div class="tier-box tier-purple">
    <h3>⏳ Dynamic ROM（瞬态）</h3>
    <p><strong>适用：</strong>瞬态动力学、CFD 非定常流、热瞬态等时间演化问题。<br>
    <strong>根据非线性强度分 4 个子类</strong>（见下表）。<br>
    <strong>输出：</strong>全场随时间演化 + 关注的标量输出（力、温度极值等）。<br>
    <strong>部署：</strong>导出为 FMU（FMI 3.0 标准）、.rom 文件或 C 代码。</p>
  </div>
</div>

<h3>Dynamic ROM 四种子类型</h3>
<table>
<tr><th>类型</th><th>算法</th><th>非线性能力</th><th>适用场景</th><th>典型加速比</th></tr>
<tr><td><strong>LTI ROM</strong><br>线性时不变</td><td>平衡截断 Balanced Truncation / Arnoldi / Krylov</td><td><span class="badge badge-red">不支持</span></td><td>结构微振动、小扰动气弹、线性热传导、RLC 电路</td><td><span class="badge badge-green">1000–10000×</span></td></tr>
<tr><td><strong>LPV ROM</strong><br>线性变参数</td><td>多组 LTI 在参数网格上插值</td><td><span class="badge badge-yellow">弱非线性</span></td><td>变工况气弹、变转速旋转机械</td><td><span class="badge badge-green">500–5000×</span></td></tr>
<tr><td><strong>POD-Galerkin + DEIM</strong><br>非线性 ROM</td><td>POD 空间压缩 → Galerkin 投影 → DEIM 非线性加速</td><td><span class="badge badge-green">强非线性</span></td><td>CFD 湍流、涡激振动、塑性变形、燃烧不稳定性</td><td><span class="badge badge-blue">100–650×</span></td></tr>
<tr><td><strong>数据驱动 ROM</strong><br>LSTM / NARX</td><td>循环神经网络 LSTM / 非线性自回归 NARX</td><td><span class="badge badge-green">强非线性</span></td><td>仅有单点传感器数据、无全场数据的时序预测</td><td><span class="badge badge-blue">100–1000×</span></td></tr>
</table>

<h3>部署与导出</h3>
<div class="grid2">
  <div class="model-card"><h4>📦 FMU（FMI 标准）</h4><p>Functional Mock-up Unit，FMI 3.0 国际标准。可嵌入 Simulink、AMESim、任何支持 FMI 的平台。<span class="badge badge-green">推荐首选</span></p></div>
  <div class="model-card"><h4>📄 .rom 文件</h4><p>Twin Builder 原生格式，包含模态基、插值矩阵、参数映射。<span class="badge badge-blue">原生格式</span></p></div>
  <div class="model-card"><h4>💻 C 代码</h4><p>生成独立 C 源码，无外部依赖。适合嵌入式部署（MCU/PLC）。<span class="badge badge-yellow">嵌入式友好</span></p></div>
  <div class="model-card"><h4>☁️ 云端/数字孪生</h4><p>ROM 部署为微服务 REST API，与 Ansys TwinAI 和物联网平台集成。<span class="badge badge-purple">DT 就绪</span></p></div>
</div>

<h2>🐍 Python 示例代码</h2>
<p>以下三段独立可运行的简化代码，展示 POD、DMD、DEIM 的核心数学逻辑。每段约 30–50 行，仅依赖 <code>numpy</code>。</p>

<h3>5.1 POD — 本征正交分解（基于 SVD）</h3>
<pre><code>#!/usr/bin/env python3
"""POD 示例：对合成流场快照做 SVD，提取模态基并重构"""
import numpy as np

# ---------- 1. 生成合成数据 ----------
nx, nt = 100, 200
x = np.linspace(0, 1, nx)
t = np.linspace(0, 5, nt)

X = np.zeros((nx, nt))
for k, (freq, decay) in enumerate([(1, 0), (2.5, 0.1), (5, 0.3)]):
    spatial = np.sin((k+1) * np.pi * x)
    temporal = np.cos(2*np.pi * freq * t) * np.exp(-decay * t)
    X += np.outer(spatial, temporal)

# ---------- 2. SVD 分解 ----------
U, s, Vt = np.linalg.svd(X, full_matrices=False)

# ---------- 3. 能量截断（保留 99% 能量）----------
energy = np.cumsum(s**2) / np.sum(s**2)
r = np.searchsorted(energy, 0.99) + 1
print(f"原始自由度: {nx} → 截断后: {r} 阶")
print(f"保留能量: {energy[r-1]*100:.2f}%")

# ---------- 4. 重构 ----------
Phi = U[:, :r]
A = np.diag(s[:r]) @ Vt[:r, :]
X_rec = Phi @ A
error = np.linalg.norm(X - X_rec) / np.linalg.norm(X)
print(f"重构相对误差: {error:.2e}")</code></pre>

<h3>5.2 DMD — 动态模态分解</h3>
<pre><code>#!/usr/bin/env python3
"""DMD 示例：从快照序列提取动力学模态（频率 + 增长/衰减率）"""
import numpy as np

nx, nt = 80, 150
x = np.linspace(0, 1, nx)
t = np.linspace(0, 10, nt)
dt = t[1] - t[0]

X = np.zeros((nx, nt))
for omega, grow in [(2.0, -0.05), (5.5, 0.02)]:
    for i in range(nx):
        X[i, :] += np.sin((i+1) * np.pi * x[i]) * np.cos(omega * t) * np.exp(grow * t)

X1 = X[:, :-1]; X2 = X[:, 1:]
U, s, Vt = np.linalg.svd(X1, full_matrices=False)
r = np.searchsorted(np.cumsum(s)/np.sum(s), 0.999) + 1
Ur, sr, Vr = U[:, :r], s[:r], Vt[:r, :].T

A_tilde = Ur.T @ X2 @ Vr @ np.diag(1/sr)
eigvals, eigvecs = np.linalg.eig(A_tilde)
phi = X2 @ Vr @ np.diag(1/sr) @ eigvecs
omega = np.log(eigvals) / dt

print("DMD 模态分析结果：")
for k in range(r):
    freq = np.abs(np.imag(omega[k])) / (2*np.pi)
    growth = np.real(omega[k])
    if freq > 0.01:
        print(f"  模态{k}: 频率={freq:.3f} Hz, 增长率={growth:+.4f}")</code></pre>

<h3>5.3 DEIM — 离散经验插值</h3>
<pre><code>#!/usr/bin/env python3
"""DEIM 示例：贪心算法选择插值点，构建非线性项加速矩阵"""
import numpy as np

N_grid, n_snaps, n_modes = 500, 200, 15
np.random.seed(42)
x = np.linspace(0, 1, N_grid)
U_f = np.zeros((N_grid, n_modes))
for k in range(n_modes):
    U_f[:, k] = np.sin((k+1) * np.pi * x) + 0.1 * np.random.randn(N_grid)

def deim_greedy(U):
    m = U.shape[1]; N = U.shape[0]
    idx = np.zeros(m, dtype=int); P = np.zeros((N, m))
    idx[0] = np.argmax(np.abs(U[:, 0]))
    P[idx[0], 0] = 1
    for j in range(1, m):
        c = np.linalg.solve(P[:, :j].T @ U[:, :j], P[:, :j].T @ U[:, j])
        r = U[:, j] - U[:, :j] @ c
        idx[j] = np.argmax(np.abs(r))
        P[idx[j], j] = 1
    return idx, P

idx, P = deim_greedy(U_f)
print(f"从 {N_grid} 个网格点中选出 {n_modes} 个插值点")
M = U_f @ np.linalg.inv(P.T @ U_f)

test_f = np.sin(3*np.pi*x) + 0.5*np.sin(7*np.pi*x)
f_selected = P.T @ test_f
f_recovered = M @ f_selected
err = np.linalg.norm(test_f - f_recovered) / np.linalg.norm(test_f)
print(f"恢复相对误差: {err:.4e}")
print(f"在线加速: O(N)={N_grid} → O(m)={n_modes}  ({N_grid//n_modes}x 每步)")</code></pre>

<h2>📋 工程选型速查</h2>
<table>
<tr><th>工程场景</th><th>推荐方法</th><th>加速比</th><th>注意事项</th></tr>
<tr><td><strong>电子散热 / 稳态热</strong></td><td>Static ROM: POD + Kriging 响应面</td><td>3000–7200×</td><td>参数空间要均匀 DOE 采样</td></tr>
<tr><td><strong>结构微振动 / 模态分析</strong></td><td>Dynamic LTI ROM: 平衡截断 / Krylov</td><td>5000–21600×</td><td>仅适用于线性假设</td></tr>
<tr><td><strong>变工况旋转机械</strong></td><td>Dynamic LPV ROM: 多组 LTI 插值</td><td>500–5000×</td><td>工况突变需验证插值过渡区</td></tr>
<tr><td><strong>CFD 涡激振动 / 湍流</strong></td><td>POD-Galerkin + DEIM</td><td>100–650×</td><td>DEIM 选点 50–90 个误差 &lt;1%</td></tr>
<tr><td><strong>塑性 / 大变形非线性</strong></td><td>POD-Galerkin + DEIM 或 LSTM</td><td>50–300×</td><td>路径依赖问题需覆盖全加载历史</td></tr>
<tr><td><strong>仅有传感器数据</strong></td><td>数据驱动 ROM: NARX / LSTM</td><td>100–1000×</td><td>仅单点时序预测，无法恢复全场</td></tr>
<tr><td><strong>设计空间探索</strong></td><td>Static ROM: POD + 响应面</td><td>1000–5000×</td><td>克里金插值比多项式精度更高</td></tr>
<tr><td><strong>实时控制 / HIL</strong></td><td>LTI ROM（首选）导出 C 代码 / FMU</td><td>10000+×</td><td>实时性要求 &lt;1ms/步</td></tr>
</table>

<h2>📚 参考资料</h2>
<div class="grid2">
  <div>
    <h3>📖 官方文档</h3>
    <ul>
      <li><a href="https://www.ansys.com/products/digital-twin/ansys-twin-builder" target="_blank">Ansys Twin Builder — Product Page</a></li>
      <li><a href="https://2021.help.ansys.com/" target="_blank">Ansys Twin Builder Help Manual</a></li>
      <li><a href="https://fmi-standard.org/" target="_blank">FMI 3.0 Standard</a></li>
    </ul>
  </div>
  <div>
    <h3>📄 学术文献</h3>
    <ul>
      <li><strong>Chaturantabut &amp; Sorensen (2010)</strong> — DEIM 原始论文. SIAM J. Sci. Comput. 32(5)</li>
      <li><strong>Stabile &amp; Rozza (2018)</strong> — POD-Galerkin NS 方程经典算例. Computers &amp; Fluids 173</li>
      <li><strong>Drmac &amp; Gugercin (2015)</strong> — Q-DEIM 改进. SIAM J. Sci. Comput. 38(2)</li>
      <li><strong>Schmid (2010)</strong> — DMD 奠基性论文. J. Fluid Mech. 656</li>
    </ul>
  </div>
</div>

<div class="tip"><strong>核心建议：</strong>线性问题直接用 LTI/LPV ROM（Twin Builder 一键构建）。非线性 CFD 走 POD-Galerkin + DEIM（需要脚本定制）。仅有传感器数据用数据驱动 ROM（LSTM/NARX）。所有 ROM 最终都导出为 FMU——工业部署的通用语言。</div>`
  },

  // ===== 5. RO-PINN Report =====
  {
    id: "ro-pinn-report",
    title: "RO-PINN 方法技术报告",
    date: "2026-07-09",
    category: "调研",
    excerpt: "POD-ROM × Physics-Informed Neural Networks 技术综述，四大学术路线对比与 Python 最小实现。",
    content: `<div class="tier-box tier-blue">
  <p style="font-size:1.05em;margin:0;"><strong>一句话定义：</strong>用 <span class="highlight">POD（Proper Orthogonal Decomposition）</span> 压缩 PDE 解空间，在降阶空间中训练 PINN，强制物理残差约束。</p>
</div>

<div class="grid2">
  <div>
    <h3>🔍 动机</h3>
    <div class="spec-row"><span class="spec-label">纯 PINN</span><span class="spec-value" style="color:var(--red)">高维训练慢 + 不稳定</span></div>
    <div class="spec-row"><span class="spec-label">纯 POD-ROM</span><span class="spec-value" style="color:var(--yellow)">非线性处理弱</span></div>
    <div class="spec-row"><span class="spec-label">RO-PINN</span><span class="spec-value" style="color:var(--green)">结合两者优势</span></div>
  </div>
  <div>
    <h3>🎯 核心优势</h3>
    <ul>
      <li>降阶空间维数 r ≪ N（通常 r=5~50, N=10⁴~10⁶）</li>
      <li>PINN 在小空间训练，速度提升 10~100×</li>
      <li>物理残差约束保证外推可靠性</li>
      <li>自然兼容参数化 PDE（μ 作为网络输入）</li>
    </ul>
  </div>
</div>

<div class="tip">离线完成 POD 分解后，在线推理只需在 r 维空间运行，速度接近传统 ROM，但保留 PINN 的非线性表达和参数化能力。</div>

<h2>📐 数学框架</h2>
<div class="grid2">
  <div>
    <h3>1. POD 分解</h3>
    <p>S = V Σ Wᵀ ——快照矩阵 S ∈ ℝ<sup>N×K</sup> 的截断 SVD，取前 r 阶模态：</p>
    <p>u(t, μ) ≈ V<sub>r</sub> · a(t, μ)</p>
    <p>V<sub>r</sub> ∈ ℝ<sup>N×r</sup>（空间 POD 基），a ∈ ℝ<sup>r</sup>（时间系数）</p>
    <div class="spec-row"><span class="spec-label">模态数 r</span><span class="spec-value">通常 5~50</span></div>
    <div class="spec-row"><span class="spec-label">能量捕获</span><span class="spec-value">通常 > 99%</span></div>
  </div>
  <div>
    <h3>2. 降阶 ODE（Galerkin 投影）</h3>
    <p>da/dt = f<sub>r</sub>(a; μ) ——将原 PDE（∂u/∂t = F(u;μ)）投影到 POD 基：</p>
    <p>f<sub>r</sub>(a;μ) = V<sub>r</sub><sup>T</sup> F(V<sub>r</sub>a; μ)</p>
  </div>
</div>

<h3>3. RO-PINN 损失函数</h3>
<div class="tier-box tier-blue">
  <p style="text-align:center;">L = L<sub>data</sub> + λ · L<sub>ROM</sub></p>
</div>
<div class="grid2">
  <div><h4>L<sub>data</sub> — 重构误差</h4><p>‖a<sub>true</sub> − NN(t, μ; θ)‖²</p></div>
  <div><h4>L<sub>ROM</sub> — 降阶 PDE 残差</h4><p>‖d(NN)/dt − f<sub>r</sub>(NN; μ)‖²</p></div>
</div>

<div class="tip"><strong>关键创新：</strong>传统 PINN 在原始物理空间计算 PDE 残差（N 维），RO-PINN 在降阶空间计算残差（r 维）。这不仅大幅减少计算量，还避免了 PINN 在高维空间中优化困难的本质问题。</div>

<h2>🔬 四大学术路线</h2>

<h3>路线一：离散化 RO-PINN</h3>
<p><span class="badge badge-blue">Halder, Stabile, Rozza (2025)</span> arXiv:2311.14045</p>
<div class="grid2">
  <div>
    <h4>方法论</h4>
    <ol><li>控制方程有限差分离散</li><li>POD-Galerkin 投影到低维空间</li><li><strong>离散残差</strong>作为 PINN 损失项</li><li>LSTM / ANN 网络学习降阶动力学</li></ol>
  </div>
  <div>
    <h4>关键发现</h4>
    <table><tr><td>网络架构</td><td>LSTM > ANN（处理时序）</td></tr><tr><td>测试案例</td><td>俯仰-沉浮翼型 + 1D 粘性 Burgers</td></tr><tr><td>核心优势</td><td>避免与外部求解器自动微分耦合</td></tr></table>
  </div>
</div>

<h3>路线二：FEM 残差 ROM-PINN</h3>
<p><span class="badge badge-cyan">Ares De Parga et al. (2025)</span> arXiv:2504.13875 · Axioms 2025</p>
<div class="grid2">
  <div>
    <h4>PROM-ANN 架构</h4>
    <ul><li><strong>PROM（Proper ROM）：</strong>POD 基 + Galerkin 投影</li><li><strong>ANN：</strong>神经网络学习降阶映射</li><li><strong>损失：</strong>直接调用 FEM 软件（Kratos）计算残差</li><li><strong>不依赖解析 PDE：</strong>黑盒求解器友好</li></ul>
  </div>
  <div>
    <h4>测试与结果</h4>
    <table><tr><td>测试案例</td><td>超弹性橡胶悬臂梁</td></tr><tr><td>非线性程度</td><td>强非线性（大变形）</td></tr><tr><td>精度提升</td><td>修正版比纯 POD 高数个数量级</td></tr></table>
  </div>
</div>

<h3>路线三：涡激振动 POD-PINN</h3>
<p><span class="badge badge-green">Yin, Janocha, Ong (2024)</span> DOI:10.1115/1.4064476 · ASME JOMAE</p>
<div class="grid2">
  <div>
    <h4>离线-在线分离架构</h4>
    <div class="spec-row"><span class="spec-label">离线</span><span class="spec-value">CFD 快照 → POD 空间模态</span></div>
    <div class="spec-row"><span class="spec-label">在线</span><span class="spec-value">PINN 学习 t → a(t)</span></div>
    <div class="spec-row"><span class="spec-label">损失</span><span class="spec-value">系数误差 + Galerkin 残差</span></div>
  </div>
  <div>
    <h4>测试配置</h4>
    <table><tr><td>物理场景</td><td>圆柱绕流涡激振动</td></tr><tr><td>流动状态</td><td>层流 + 湍流</td></tr><tr><td>长期稳定性</td><td>优于纯 PINN 和纯数据驱动</td></tr></table>
  </div>
</div>

<h3>路线四：行人流降阶 PINN</h3>
<p><strong>Pan, Xiao, Shen (2024)</strong> Transportation Research Part C, Vol. 163</p>
<p>首次将 PINN 框架用于求解宏观行人流模型。提出 ro-PINN——通过分解高阶 PDE、降低自动微分的导数阶数来提升性能。对比了三种方案：vanilla PINN、extended-variable PINN（ev-PINN）、ro-PINN。</p>

<h2>📊 方法对比表</h2>
<table>
<tr><th>方法</th><th>物理约束</th><th>非线性能力</th><th>在线速度</th><th>论文</th></tr>
<tr><td><strong>纯 POD-Galerkin</strong></td><td><span class="badge badge-green">强</span></td><td><span class="badge badge-red">弱</span></td><td><span class="badge badge-green">极快</span></td><td>传统</td></tr>
<tr><td><strong>纯 PINN</strong></td><td><span class="badge badge-green">强</span></td><td><span class="badge badge-green">强</span></td><td><span class="badge badge-red">慢</span></td><td>Raissi 2019</td></tr>
<tr><td><strong>RO-PINN (Halder)</strong></td><td><span class="badge badge-yellow">中</span></td><td><span class="badge badge-yellow">中</span></td><td><span class="badge badge-green">快</span></td><td>2025</td></tr>
<tr><td><strong>FEM-PINN-ROM (Ares)</strong></td><td><span class="badge badge-green">强</span></td><td><span class="badge badge-green">强</span></td><td><span class="badge badge-yellow">中</span></td><td>2025</td></tr>
<tr><td><strong>POD-PINN (Yin)</strong></td><td><span class="badge badge-yellow">中</span></td><td><span class="badge badge-yellow">中</span></td><td><span class="badge badge-green">快</span></td><td>2024</td></tr>
<tr><td><strong>ro-PINN (Pan/Xiao/Shen)</strong></td><td><span class="badge badge-green">中</span></td><td><span class="badge badge-green">中</span></td><td><span class="badge badge-green">快</span></td><td>2024</td></tr>
</table>

<h2>🚀 前沿拓展</h2>
<div class="grid3">
  <div class="model-card"><h4>🧬 Neural-POD</h4><p>用神经网络学习非线性正交基，替代传统线性 POD。未来方向：引入 PINN 物理约束基函数。<span class="badge badge-cyan">基函数学习</span></p></div>
  <div class="model-card"><h4>🔄 CNF-ROM</h4><p>条件归一化流 ROM + PINN 微调。利用 flow 的分布建模能力做参数外推。<span class="badge badge-blue">生成式 ROM</span></p></div>
  <div class="model-card"><h4>⚖️ 自校准权重</h4><p>自动平衡 L<sub>data</sub> 和 L<sub>physics</sub> 的权重 λ。避免手动调参，提升训练稳定性和泛化能力。<span class="badge badge-yellow">训练优化</span></p></div>
</div>

<h2>🐍 Python 最小示例</h2>
<p>下方代码实现 RO-PINN 简化核心流程：1D Burgers 方程 + POD 降阶 + MLP 网络 + 物理残差约束。</p>

<pre><code>#!/usr/bin/env python3
"""RO-PINN 最小示例：1D Burgers 方程 + POD 降阶 + 物理约束"""
import numpy as np
import torch
import torch.nn as nn

# ─── 1. 生成合成快照数据（1D Burgers 有限差分解）───
def burgers_fd(nu=0.01, nx=128, nt=200):
    dx = 2.0 / (nx - 1); dt = 0.005
    x = np.linspace(-1, 1, nx)
    u = -np.sin(np.pi * x)
    snapshots = [u.copy()]
    for _ in range(nt - 1):
        un = u.copy()
        for i in range(1, nx - 1):
            u[i] = un[i] - un[i]*dt/(2*dx)*(un[i+1]-un[i-1]) \\
                   + nu*dt/dx**2*(un[i+1]-2*un[i]+un[i-1])
        snapshots.append(u.copy())
    return x, np.linspace(0, (nt-1)*dt, nt), np.array(snapshots).T

# ─── 2. POD/SVD 模态提取 ───
def pod_decompose(S, r=5):
    V, sigma, Wt = np.linalg.svd(S, full_matrices=False)
    V_r = V[:, :r]
    a_coeff = np.diag(sigma[:r]) @ Wt[:r, :]
    return V_r, a_coeff, sigma

# ─── 3. 神经网络模型 ───
class RO_PINN_Net(nn.Module):
    def __init__(self, r=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, r)
        )

    def forward(self, t, nu):
        inp = torch.stack([t, nu.expand_as(t)], dim=-1)
        return self.net(inp)

# ─── 4. 降阶残差（Burgers Galerkin 投影）───
def reduced_residual(a, V_r, dx, nu):
    nx = V_r.shape[0]
    u = V_r @ a.numpy().T
    D1 = (np.eye(nx, k=1) - np.eye(nx, k=-1)) / (2*dx)
    D2 = (np.eye(nx, k=1) - 2*np.eye(nx) + np.eye(nx, k=-1)) / dx**2
    u_x = D1 @ u; u_xx = D2 @ u
    residual = -u * u_x + nu * u_xx
    f_r = V_r.T @ residual
    return torch.tensor(f_r.T, dtype=torch.float32)

# ─── 5. 主训练流程 ───
nu = 0.01; r = 5; λ = 0.1; epochs = 5000
x, t_vals, S = burgers_fd(nu=nu)
V_r, a_coeff, sigma = pod_decompose(S, r=r)
dx = x[1] - x[0]
print(f"POD r={r}, 能量捕获: {np.sum(sigma[:r]**2)/np.sum(sigma**2)*100:.2f}%")

# 对比训练：Pure Data vs RO-PINN
for name, use_physics in [("Pure Data", False), ("RO-PINN", True)]:
    model = RO_PINN_Net(r=r)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        a_pred = model(t_train, nu_tensor)
        loss_data = nn.functional.mse_loss(a_pred, a_train)
        if use_physics:
            da_dt = torch.autograd.grad(a_pred, t_train,
                torch.ones_like(a_pred), create_graph=True)[0]
            f_r = reduced_residual(a_pred, V_r, dx, nu)
            loss_rom = nn.functional.mse_loss(da_dt, f_r)
            loss = loss_data + λ * loss_rom
        else:
            loss = loss_data
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"{name}: Test MSE = {test_err:.2e}")

print("✓ RO-PINN 示例完成：POD 降阶 + 物理残差约束已集成")</code></pre>

<h2>📚 论文原文链接</h2>
<table>
<tr><th>论文</th><th>链接</th></tr>
<tr><td><strong>Halder, Stabile, Rozza (2025)</strong> — Discrete RO-PINNs</td><td><a href="https://arxiv.org/abs/2311.14045" target="_blank">arxiv.org/abs/2311.14045</a></td></tr>
<tr><td><strong>Ares De Parga et al. (2025)</strong> — PROM-ANN</td><td><a href="https://arxiv.org/abs/2504.13875" target="_blank">arxiv.org/abs/2504.13875</a></td></tr>
<tr><td><strong>Yin, Janocha, Ong (2024)</strong> — VIV POD-PINN</td><td><a href="https://doi.org/10.1115/1.4064476" target="_blank">doi.org/10.1115/1.4064476</a></td></tr>
<tr><td><strong>Pan, Xiao, Shen (2024)</strong> — 行人流降阶 PINN</td><td><a href="https://doi.org/10.1016/j.trc.2024.104658" target="_blank">doi.org/10.1016/j.trc.2024.104658</a></td></tr>
<tr><td><strong>Raissi et al. (2019)</strong> — PINN 奠基论文</td><td><a href="https://doi.org/10.1016/j.jcp.2018.10.045" target="_blank">doi.org/10.1016/j.jcp.2018.10.045</a></td></tr>
<tr><td><strong>GitHub 开源实现</strong></td><td><a href="https://github.com/biromiro/pod-pinn" target="_blank">github.com/biromiro/pod-pinn</a></td></tr>
</table>

<div class="tier-box tier-blue">
  <h3>RO-PINN 方法论定位</h3>
  <p>RO-PINN 不是单一算法，而是一类将 POD 降阶与 PINN 物理约束相结合的混合策略。不同路线根据残差来源（离散/FEM/Galerkin）和网络架构（ANN/LSTM/条件流）产生差异化优势，但它们共享一个核心洞察：<strong>在低维空间用神经网络逼近降阶动力学，比在高维空间直接求解 PDE 更高效、更稳定。</strong></p>
</div>

<h2>📌 选型建议</h2>
<table>
<tr><th>场景</th><th>推荐方法</th><th>理由</th></tr>
<tr><td>有现成 CFD/FEM 求解器</td><td>FEM-PINN-ROM (Ares)</td><td>直接调用黑盒残差，无需解析 PDE</td></tr>
<tr><td>参数化 PDE，快速在线推理</td><td>RO-PINN (Halder)</td><td>LSTM + 离散残差，输入灵活</td></tr>
<tr><td>流固耦合长期预测</td><td>POD-PINN (Yin)</td><td>离线-在线分离，长期稳定性好</td></tr>
<tr><td>前沿研究 / 非线性基</td><td>Neural-POD</td><td>可学习基函数 + PINN 约束</td></tr>
<tr><td>不确定性量化 / 外推</td><td>CNF-ROM</td><td>生成模型 + 精确似然</td></tr>
</table>

<div class="tip"><strong>核心脉络：</strong>POD 提供空间压缩（N→r），PINN 提供降阶动力学学习 + 物理正则化。2023–2025 的发展趋势是从"POD+PINN 简单拼接"走向"基函数学习和物理约束的深度融合"。</div>`
  },

  // ===== 6. SD Models Guide 3060Ti 8GB =====
  {
    id: "sd-models-guide-3060ti",
    title: "Stable Diffusion 模型全景指南",
    date: "2026-07-08",
    category: "工程实践",
    excerpt: "RTX 3060 Ti 8GB 专用 Stable Diffusion 模型选型指南，涵盖写实、二次元、2.5D、场景、特殊风格及显存优化。",
    content: `<h2>🖥️ 你的显卡：能力与天花板</h2>
<div class="grid2">
  <div>
    <div class="spec-row"><span class="spec-label">型号</span><span class="spec-value">NVIDIA RTX 3060 Ti</span></div>
    <div class="spec-row"><span class="spec-label">VRAM</span><span class="spec-value" style="color:var(--yellow)">8 GB GDDR6</span></div>
    <div class="spec-row"><span class="spec-label">CUDA 核心</span><span class="spec-value">4,864</span></div>
    <div class="spec-row"><span class="spec-label">带宽</span><span class="spec-value">448 GB/s</span></div>
  </div>
  <div>
    <div class="spec-row"><span class="spec-label">SD 1.5 (512px)</span><span class="spec-value" style="color:var(--green)">~6 it/s ⚡</span></div>
    <div class="spec-row"><span class="spec-label">SDXL (1024px)</span><span class="spec-value" style="color:var(--yellow)">~1-2 it/s</span></div>
    <div class="spec-row"><span class="spec-label">Flux (NF4量化)</span><span class="spec-value" style="color:var(--orange)">2-3 分钟/张</span></div>
    <div class="spec-row"><span class="spec-label">核心优势</span><span class="spec-value" style="color:var(--green)">算力够用，速度快</span></div>
  </div>
</div>

<h3>VRAM 占用参考（8GB 上限）</h3>
<div class="spec-row"><span class="spec-label">SD 1.5 (FP16)</span><span class="spec-value" style="color:var(--green)">~2.5 GB</span></div>
<div class="vram-bar"><div class="vram-fill safe" style="width:31%"></div></div>
<div class="spec-row"><span class="spec-label">SDXL (FP16)</span><span class="spec-value" style="color:var(--yellow)">~5.5 GB</span></div>
<div class="vram-bar"><div class="vram-fill warn" style="width:69%"></div></div>
<div class="spec-row"><span class="spec-label">SDXL + ControlNet</span><span class="spec-value" style="color:var(--orange)">~6.5-7.5 GB</span></div>
<div class="vram-bar"><div class="vram-fill warn" style="width:88%"></div></div>
<div class="spec-row"><span class="spec-label">Flux Dev (FP16)</span><span class="spec-value" style="color:var(--red)">~14 GB ❌</span></div>
<div class="vram-bar"><div class="vram-fill danger" style="width:100%"></div></div>
<div class="spec-row"><span class="spec-label">Flux Dev (NF4量化)</span><span class="spec-value" style="color:var(--yellow)">~6-7 GB</span></div>
<div class="vram-bar"><div class="vram-fill warn" style="width:81%"></div></div>

<div class="tip">一句话总结：算力够快，显存是短板。SD 1.5 随便跑，SDXL 需要优化，Flux 只能走量化。</div>

<h2>🏗️ 四大架构速览</h2>
<table>
<tr><th>架构</th><th>年份</th><th>分辨率</th><th>显存</th><th>速度</th><th>生态</th><th>体验</th></tr>
<tr><td><strong>SD 1.5</strong></td><td>2022</td><td>512×512</td><td>~2.5 GB</td><td>6 it/s</td><td>海量</td><td>🟢 完美</td></tr>
<tr><td><strong>SDXL / Lightning</strong></td><td>2023</td><td>1024×1024</td><td>~5.5 GB</td><td>1-2 it/s</td><td>极其成熟</td><td>🟡 优化可用</td></tr>
<tr><td><strong>SD 3.5</strong></td><td>2024</td><td>1024×1024</td><td>~6-7.5 GB</td><td>一般</td><td>还在长</td><td>🟡 临界</td></tr>
<tr><td><strong>Flux / Schnell</strong></td><td>2024</td><td>1024×1024</td><td>6-7 GB(NF4)</td><td>很慢</td><td>较少</td><td>🔴 勉强</td></tr>
</table>

<div class="tip">对 8GB 显存来说：SD 1.5 是舒适区，SDXL Lightning 是甜点区，Flux 是挑战区。</div>

<h2>📸 写实 / 摄影风格</h2>
<table>
<tr><th>模型</th><th>架构</th><th>亮点</th><th>VRAM</th><th>推荐度</th></tr>
<tr><td><strong>Realistic Vision V6.0</strong></td><td>SD 1.5</td><td>SD1.5 最强写实人脸，低配首选</td><td>~2.5 GB</td><td>⭐⭐⭐⭐⭐</td></tr>
<tr><td><strong>RealVisXL V4.0</strong></td><td>SDXL</td><td>2025 写实天花板，皮肤质感/光影极佳</td><td>~5.5 GB</td><td>⭐⭐⭐⭐⭐</td></tr>
<tr><td><strong>Juggernaut XL v10</strong></td><td>SDXL</td><td>电影感/单反风格，构图出色</td><td>~5.5 GB</td><td>⭐⭐⭐⭐⭐</td></tr>
<tr><td><strong>CyberRealistic XL v9</strong></td><td>SDXL</td><td>复杂场景叙事力强，动作/环境真实</td><td>~5.5 GB</td><td>⭐⭐⭐⭐</td></tr>
<tr><td><strong>DreamShaper 8</strong></td><td>SD 1.5</td><td>万金油，写实+艺术都有好表现</td><td>~2.5 GB</td><td>⭐⭐⭐⭐</td></tr>
<tr><td><strong>Deliberate 3</strong></td><td>SD 1.5</td><td>照片和艺术插画中间地带</td><td>~2.5 GB</td><td>⭐⭐⭐⭐</td></tr>
<tr><td><strong>Flux Raw (NF4)</strong></td><td>Flux</td><td>极致写实，摄影级光影和皮肤</td><td>~6-7 GB</td><td>⭐⭐⭐ (慢)</td></tr>
</table>

<h2>🎨 二次元 / 动漫风格</h2>
<table>
<tr><th>模型</th><th>架构</th><th>亮点</th><th>VRAM</th><th>推荐度</th></tr>
<tr><td><strong>Anything V5 Prt</strong></td><td>SD 1.5</td><td>经典万能二次元，人物/场景/动物全搞定</td><td>~2.5 GB</td><td>⭐⭐⭐⭐⭐</td></tr>
<tr><td><strong>Counterfeit V3.0</strong></td><td>SD 1.5</td><td>细节还原度高，色彩鲜艳，壁纸级</td><td>~2.5 GB</td><td>⭐⭐⭐⭐⭐</td></tr>
<tr><td><strong>Pony Diffusion V6 XL</strong></td><td>SDXL</td><td>SDXL 最强二次元，LoRA 生态无敌，tag 控制精准</td><td>~5.5 GB</td><td>⭐⭐⭐⭐⭐</td></tr>
<tr><td><strong>Illustrious XL</strong></td><td>SDXL</td><td>2025 新贵，线条干净，手部/解剖更好</td><td>~5.5 GB</td><td>⭐⭐⭐⭐⭐</td></tr>
<tr><td><strong>NoobAI XL</strong></td><td>SDXL</td><td>VTuber/现代日系审美精准，社区活跃</td><td>~5.5 GB</td><td>⭐⭐⭐⭐</td></tr>
<tr><td><strong>AWPainting V1.4</strong></td><td>SD 1.5</td><td>国内 LiblibAI 热门，张力和色彩强</td><td>~2.5 GB</td><td>⭐⭐⭐⭐</td></tr>
<tr><td><strong>AbyssOrangeMix3 (AOM3)</strong></td><td>SD 1.5</td><td>深渊橘，光影氛围感极强</td><td>~2.5 GB</td><td>⭐⭐⭐⭐</td></tr>
</table>

<div class="grid2">
  <div class="tip">SD 1.5 党：Anything V5 + Counterfeit 双持，基本覆盖所有二次元需求。</div>
  <div class="tip">SDXL 党：Pony Diffusion + Illustrious，LoRA 生态和画质都是顶配。</div>
</div>

<h2>🎭 2.5D / 半写实风格</h2>
<div class="grid3">
  <div class="model-card"><h4>majicMIX fantasy</h4><p>SD 1.5 · 奇幻风格 2.5D 标杆，国风/古风效果拔群 ~2.5 GB</p></div>
  <div class="model-card"><h4>Never Ending Dream</h4><p>SD 1.5 · 梦境感，柔光朦胧的 2.5D 人像 ~2.5 GB</p></div>
  <div class="model-card"><h4>RealCartoon3D</h4><p>SD 1.5 · 3D 动画电影风格，接近皮克斯/迪士尼质感 ~2.5 GB</p></div>
  <div class="model-card"><h4>SDVN7-NijiStyleXL</h4><p>SDXL · 动漫+素描+写实三合一 ~5.5 GB</p></div>
  <div class="model-card"><h4>GuoFeng3 (国风)</h4><p>SD 1.5 · 中国古风特化，汉服/武侠/仙侠 ~2.5 GB</p></div>
  <div class="model-card"><h4>Pony + 写实 LoRA</h4><p>SDXL · Pony 底模加写实 LoRA，2.5D 好底子 ~5.5 GB</p></div>
</div>

<h2>🏙️ 场景 / 建筑 / 风景</h2>
<div class="grid2">
  <div>
    <h3>🏛️ 建筑 & 室内</h3>
    <table>
      <tr><th>模型</th><th>架构</th><th>VRAM</th></tr>
      <tr><td>ArchitectureRealMix</td><td>SD 1.5</td><td>~2.5 GB</td></tr>
      <tr><td>InteriorDesignSuperMix</td><td>SD 1.5</td><td>~2.5 GB</td></tr>
      <tr><td>RealVisXL + ControlNet Depth</td><td>SDXL</td><td>~6.5 GB</td></tr>
    </table>
  </div>
  <div>
    <h3>🌄 风景 & 自然</h3>
    <table>
      <tr><th>模型</th><th>架构</th><th>VRAM</th></tr>
      <tr><td>DreamShaper 8</td><td>SD 1.5</td><td>~2.5 GB</td></tr>
      <tr><td>Juggernaut XL</td><td>SDXL</td><td>~5.5 GB</td></tr>
      <tr><td>Flux Dev (NF4)</td><td>Flux</td><td>~6-7 GB</td></tr>
    </table>
  </div>
</div>

<h2>🎯 特殊风格</h2>
<div class="grid2">
  <div class="model-card"><h4>🎬 SDXL Lightning (4步)</h4><p>蒸馏版 SDXL，只需 4 步采样，速度快 3-5 倍。8GB 跑 SDXL 的终极方案。~5 GB · 极快</p></div>
  <div class="model-card"><h4>⚡ SDXL Turbo (1步)</h4><p>一步出图，速度最快，但有画质折损。适合快速原型/草稿。~5 GB · 极快</p></div>
  <div class="model-card"><h4>📝 SD 3.5 Medium</h4><p>写文字最强的 SD 系列，做海报/logo/文字设计推荐。~6-7.5 GB</p></div>
  <div class="model-card"><h4>🎨 Flux Schnell (NF4)</h4><p>Flux 快速版，4 步出图，比 Dev 快很多。8GB 跑 Flux 最友好的选择。~6 GB · 较快</p></div>
  <div class="model-card"><h4>🖼️ Z-Image Turbo</h4><p>字节跳动 S3-DiT 架构，8 步推理。需要 10GB+ VRAM，临界。~8+ GB · 临界</p></div>
  <div class="model-card"><h4>🎥 AnimateDiff</h4><p>AI 视频/动画生成。多帧（16帧）需要 14-18GB，你的卡基本不行。14+ GB · 不行</p></div>
</div>

<h2>📊 显存分级总览</h2>
<div class="tier-box tier-green">
  <h3>🟢 绿色区：随便跑 (~2-4 GB)</h3>
  <p>SD 1.5 全家桶：Realistic Vision / DreamShaper / Anything V5 / Counterfeit / Deliberate / AWPainting / majicMIX / ChilloutMix 等</p>
</div>
<div class="tier-box tier-yellow">
  <h3>🟡 黄色区：优化可跑 (~5-7 GB)</h3>
  <p>SDXL 全家桶：RealVisXL / Juggernaut XL / Pony Diffusion / Illustrious / NoobAI / AAM AnimeMix / CyberRealistic XL / SDXL Lightning · SD 3.5 Medium · Flux Schnell (NF4)</p>
</div>
<div class="tier-box tier-red">
  <h3>🔴 红色区：艰难或不行 (7+ GB)</h3>
  <p>Flux Dev (NF4) · Z-Image Turbo · SD 3.5 Large · SDXL + 多个 ControlNet · AnimateDiff 视频 · 模型训练</p>
</div>

<h2>🔧 8GB 优化手册</h2>
<div class="grid2">
  <div>
    <h3>启动参数（必加）</h3>
    <pre><code># A1111 / Forge
--medvram
--xformers
--opt-split-attention
--no-half-vae

# ComfyUI
--lowvram (SDXL)
--normalvram (SD 1.5)</code></pre>
    <h3>UI 内设置</h3>
    <table>
      <tr><td>Batch size</td><td style="color:var(--red)">必须 = 1</td></tr>
      <tr><td>精度</td><td>FP16 / BF16</td></tr>
      <tr><td>Tiled VAE</td><td style="color:var(--green)">开启</td></tr>
      <tr><td>Attention 切片</td><td style="color:var(--green)">开启</td></tr>
      <tr><td>CPU Offload</td><td>仅 Flux NF4 时</td></tr>
    </table>
  </div>
  <div>
    <h3>模型格式选择</h3>
    <p>FP32(全量) → FP16(省一半) → BF16(更稳定) → INT8(略有损) → NF4/GGUF(Flux 专用)</p>
    <h3>推荐工具链</h3>
    <p><span class="badge badge-green">ComfyUI</span> 最省显存，节点式工作流<br>
    <span class="badge badge-green">Forge (WebUI)</span> 性价比王，SD 1.5/SDXL 最优解<br>
    <span class="badge badge-yellow">Automatic1111</span> 经典但费内存</p>
  </div>
</div>

<div class="warning">别同时开 Chrome 几十个标签页 + Discord + 游戏客户端跑 SD。浏览器 GPU 加速也会吃显存。</div>

<h2>🍳 快速上手食谱</h2>
<div class="grid2">
  <div><h3>🥇 写实人像配方</h3><ol><li>下载 RealVisXL V4.0（CivitAI）</li><li>Forge 启动加 --medvram</li><li>分辨率 1024×1024 / 采样 DPM++ 2M</li><li>加个 Lightning LoRA 提速</li><li>出图后用 hires.fix 放大到 2K</li></ol></div>
  <div><h3>🥈 二次元角色配方</h3><ol><li>SD 1.5 用 Anything V5（省显存快）</li><li>SDXL 用 Pony Diffusion V6</li><li>去 CivitAI 找对应角色 LoRA</li><li>正面 prompt 加 quality tags</li><li>负面 prompt: EasyNegative embedding</li></ol></div>
  <div><h3>🥉 建筑/场景配方</h3><ol><li>底模用 RealVisXL 或 Juggernaut XL</li><li>ControlNet Depth 丢一张草稿/照片</li><li>ControlNet 权重 0.6-0.8</li><li>配合建筑/室内 LoRA 微调风格</li><li>记得开 Tiled VAE 防 OOM</li></ol></div>
  <div><h3>🏅 Flux 尝鲜配方</h3><ol><li>ComfyUI 装 Flux Dev NF4 版本</li><li>加载 GGUF Q4_K_M 量化模型</li><li>单图预计 2-3 分钟，耐心等</li><li>prompt 尽量用自然语言描述</li><li>Schnell 版更快但画质一般</li></ol></div>
</div>

<h2>📥 去哪下载</h2>
<div class="grid2">
  <div class="model-card"><h4>CivitAI</h4><p>全球最大 SD 模型社区。排行榜筛选、点赞数、示例图一应俱全。</p></div>
  <div class="model-card"><h4>HuggingFace</h4><p>官方模型发布地。SD 1.5/SDXL/Flux 原版 + NF4/GGUF 量化版。</p></div>
  <div class="model-card"><h4>LiblibAI</h4><p>国内平台，下载速度快。AWPainting、GuoFeng3 等国人模型首发地。</p></div>
  <div class="model-card"><h4>吐司网</h4><p>国内社区，下载方便，中文界面友好。</p></div>
</div>

<h2>📌 总结</h2>
<table>
<tr><th>场景</th><th>推荐组合</th><th>体验</th></tr>
<tr><td>日常瞎玩、快速出图</td><td>Forge + SD 1.5（Anything V5 / Realistic Vision）</td><td><span class="badge badge-green">流畅</span></td></tr>
<tr><td>认真出片、追求画质</td><td>Forge + SDXL Lightning（RealVisXL / Juggernaut / Pony）</td><td><span class="badge badge-yellow">还行</span></td></tr>
<tr><td>二次元角色创作</td><td>Forge + Pony Diffusion V6 + 角色 LoRA</td><td><span class="badge badge-yellow">还行</span></td></tr>
<tr><td>建筑可视化</td><td>Forge + RealVisXL + ControlNet Depth</td><td><span class="badge badge-yellow">临界</span></td></tr>
<tr><td>极致画质、不赶时间</td><td>ComfyUI + Flux Dev NF4</td><td><span class="badge badge-red">慢</span></td></tr>
<tr><td>AI 视频 / 动画</td><td>❌ 在现有设备上不可行</td><td><span class="badge badge-red">不行</span></td></tr>
</table>

<div class="tip"><strong>核心原则：SD 1.5 保底，SDXL Lightning 主战，Flux NF4 尝鲜。</strong>你的 3060 Ti 算力不差，败在显存。善用 Lightning/Turbo 蒸馏模型和量化，8GB 也能玩得很舒服。</div>`
  }

];
