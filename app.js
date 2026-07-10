/* =============================================
   Blog Posts Data (from ShareFiles)
   ============================================= */
const blogPosts = [
  {
    id: 'paddlescience-survey',
    title: 'PaddleScience 调研报告',
    date: '2026-07-06',
    category: '调研',
    excerpt: '国产深度学习科学计算框架的全景调研：架构设计、核心能力、框架对比与选型建议。',
    content: `
      <p>PaddleScience 是百度飞桨生态下的科学计算套件，利用深度神经网络和自动微分机制解决 PDE/ODE 问题。采用 Apache 2.0 协议，是目前国内最完整的 AI4S 开源框架。</p>

      <h2>三种求解范式</h2>
      <table>
        <tr><th>范式</th><th>说明</th><th>适用场景</th></tr>
        <tr><td>物理机理驱动</td><td>PDE 作为损失函数约束，无监督/半监督</td><td>边界条件明确，监督数据稀缺</td></tr>
        <tr><td>数据驱动</td><td>纯监督学习，从数据中学习映射</td><td>有大量模拟/实验数据</td></tr>
        <tr><td>数理融合</td><td>同时使用 PDE 约束 + 数据约束</td><td>部分物理规律已知，部分数据可用</td></tr>
      </table>

      <h2>核心模块架构</h2>
      <p>PaddleScience 将求解流程拆分为 12+ 个核心模块，每个模块职责单一、可替换：</p>
      <ul>
        <li><strong>Arch</strong>：神经网络模型（MLP、Transformer、GNN 等）</li>
        <li><strong>Equation</strong>：基于 SymPy 的符号方程定义，支持联立方程组</li>
        <li><strong>Constraint</strong>：统一抽象——将 PDE、边界、监督约束统一为"约束"，使机理驱动和数据驱动共享训练流程</li>
        <li><strong>Geometry</strong>：几何采样，支持 STL 复杂几何加载与布尔运算</li>
        <li><strong>Solver</strong>：全局调度器，负责训练、评估、可视化的流程编排</li>
        <li><strong>Loss</strong>：多种损失函数 + 多任务学习权重策略（NTK、GradNorm）</li>
        <li><strong>Optimizer</strong>：优化器 + 学习率调度器</li>
      </ul>

      <h2>框架对比</h2>
      <table>
        <tr><th>维度</th><th>PaddleScience</th><th>NVIDIA Modulus</th><th>DeepXDE</th></tr>
        <tr><td>底层框架</td><td>PaddlePaddle</td><td>PyTorch</td><td>TF/PyTorch/JAX/Paddle</td></tr>
        <tr><td>PINN 支持</td><td>✅ 完善</td><td>✅ 完善</td><td>✅ 核心功能</td></tr>
        <tr><td>符号方程</td><td>✅ SymPy 原生</td><td>⚠️ 需手写</td><td>⚠️ 需手写</td></tr>
        <tr><td>几何处理</td><td>✅ STL + SymPy</td><td>✅ CSG + STL</td><td>⚠️ 简单几何</td></tr>
        <tr><td>多硬件</td><td>CUDA/XPU(昆仑)/CPU</td><td>CUDA only</td><td>CPU/CUDA</td></tr>
        <tr><td>中文文档</td><td>✅ 完善</td><td>⚠️ 英文为主</td><td>⚠️ 英文为主</td></tr>
        <tr><td>工业部署</td><td>Paddle Inference/Serving</td><td>TensorRT/Triton</td><td>学术为主</td></tr>
      </table>

      <h2>技术亮点</h2>
      <ul>
        <li><strong>Constraint 统一抽象</strong>：物理约束和数据约束共用训练流程，数理融合场景极方便</li>
        <li><strong>实验工程化</strong>：Hydra 配置管理、实验追踪、并行超参搜索，适合大规模消融实验</li>
        <li><strong>国产化适配</strong>：昆仑芯 XPU 支持，适合国产硬件环境</li>
        <li><strong>案例丰富</strong>：覆盖数学、流体、结构、传热、气象、材料、化学等 60+ 案例</li>
      </ul>

      <h2>选型建议</h2>
      <p>如果团队技术栈是 PaddlePaddle 或愿意切换，PaddleScience 是国产最优选。如果已有 PyTorch 沉淀，Modulus/PhysicsNeMo 更顺滑。建议用 PaddleScience 跑一个 Allen-Cahn 或传热案例，直观感受开发效率后再做决策。</p>

      <blockquote>PaddleScience 的核心取舍：国产生态亲和力 vs PyTorch 生态兼容性。适合有国产化需求、中文支持需求的团队。</blockquote>
    `
  },
  {
    id: 'framework-survey',
    title: '物理信息机器学习框架全景调研',
    date: '2026-07-06',
    category: '调研',
    excerpt: 'DeepXDE、PaddleScience、NVIDIA PhysicsNeMo 三大框架深度对比，以及 DMD/POD/DEIM 降阶方法梳理。',
    content: `
      <p>物理信息机器学习的核心路线：将物理规律（PDE、边界条件、守恒律）嵌入神经网络训练过程，使模型在数据稀缺时仍保持物理一致性。本文对比三大主流框架，并梳理降阶加速方法。</p>

      <h2>主流范式</h2>
      <ul>
        <li><strong>物理信息驱动的 PINN</strong>：将 PDE 残差作为损失函数，无需标注数据即可求解</li>
        <li><strong>数据驱动的神经算子</strong>：从仿真数据中学习 PDE 算子映射（DeepONet、FNO）</li>
        <li><strong>数理融合</strong>：PINN + 数据监督的混合策略</li>
        <li><strong>降维加速</strong>：POD/DMD 提取低维模态 + DEIM 加速非线性项</li>
      </ul>

      <h2>三大框架对比</h2>
      <table>
        <tr><th>维度</th><th>DeepXDE</th><th>PaddleScience</th><th>PhysicsNeMo</th></tr>
        <tr><td>开发方</td><td>布朗大学 Lu Lu 课题组</td><td>百度</td><td>NVIDIA</td></tr>
        <tr><td>底层框架</td><td>TF/PyTorch/JAX/Paddle</td><td>PaddlePaddle</td><td>PyTorch</td></tr>
        <tr><td>核心能力</td><td>PINN + DeepONet</td><td>PINN + 数据驱动 + 算子</td><td>PINN + 算子 + GNN + 扩散</td></tr>
        <tr><td>3D 几何</td><td>⚠️ 基础支持</td><td>✅ pymesh/open3d</td><td>✅ 点云/mesh 管道</td></tr>
        <tr><td>分布式训练</td><td>⚠️ 基础支持</td><td>⚠️ 单机为主</td><td>✅ 多GPU/多节点</td></tr>
        <tr><td>预训练模型</td><td>❌</td><td>❌</td><td>✅ NGC 模型库</td></tr>
        <tr><td>生态定位</td><td>学术研究</td><td>国产框架、工业验证</td><td>工业级数字孪生</td></tr>
      </table>

      <h2>DeepXDE 要点</h2>
      <p>布朗大学 Lu Lu（陆路）课题组开发，是首个提出 DeepONet 的团队。最大特色是多后端设计——通过环境变量切换 TensorFlow/PyTorch/JAX/Paddle 后端，对学术研究非常灵活。</p>
      <p>功能相对简洁，对数据驱动的大模型缺乏支持。适合 PINN 论文复现和快速原型验证。</p>

      <h2>PaddleScience 要点</h2>
      <p>通过 <code>arch</code>、<code>constraint</code>、<code>equation</code> 等模块简化了网络构造和 PDE 定义。支持三种求解范式：无监督 PINN、数据驱动监督学习、神经算子学习。3D 几何处理能力突出，依赖 open3d + pyvista + pysdf + pymesh 技术栈。</p>

      <h2>PhysicsNeMo 要点</h2>
      <p>NVIDIA 开源的大规模物理 AI 框架（前身 Modulus），基于 PyTorch。核心优势：原生 GPU 优化、多 GPU/多节点分布式训练、NGC 预训练模型库。已被 Ansys、SimScale、台积电、富士康等企业采用。适合工业级数字孪生产品化方向。</p>

      <h2>降阶加速方法</h2>
      <table>
        <tr><th>方法</th><th>本质</th><th>典型应用</th></tr>
        <tr><td>POD</td><td>纯空间统计降维，按能量排序提取模态</td><td>ROM 基函数构建</td></tr>
        <tr><td>DMD</td><td>时空耦合动力学分解，单频模态+增长/衰减率</td><td>流场稳定性分析、短期预测</td></tr>
        <tr><td>DEIM</td><td>非线性项插值加速，挑选关键插值点</td><td>配合 POD/DMD 加速 ROM</td></tr>
        <tr><td>NARX</td><td>时序神经网络预测（历史输出+外部输入）</td><td>工业传感器单点预报</td></tr>
      </table>

      <h2>选型路线图</h2>
      <ul>
        <li>学术研究、PINN 探索 → <strong>DeepXDE</strong>（轻量、多后端、论文复现友好）</li>
        <li>国内部署、复杂 3D 几何 + PINN → <strong>PaddleScience</strong>（国产生态、3D 能力强）</li>
        <li>工业级数字孪生、大规模部署 → <strong>PhysicsNeMo</strong>（GPU 加速、分布式）</li>
        <li>瞬态仿真加速 → <strong>POD/DMD + DEIM</strong>（经典 ROM 路线，可与上述框架结合）</li>
      </ul>
    `
  },
  {
    id: 'dmd-pod-deim',
    title: 'DMD、POD 与 DEIM 降阶方法详解',
    date: '2026-06-25',
    category: '技术笔记',
    excerpt: '动态模态分解、本征正交分解与离散经验插值法的原理对比、适用场景与工程应用指南。',
    content: `
      <p>DMD、POD、DEIM 是仿真降阶加速的三大核心技术。本文从直观原理出发，对比三者差异，给出工程选型建议。</p>

      <h2>POD：本征正交分解</h2>
      <p><strong>核心思想</strong>：与 PCA 类似，对全场快照数据做 SVD 分解，按能量大小排序提取空间模态。少量模态即可保留绝大部分能量，常用作 ROM 的基函数。</p>

      <h3>特点</h3>
      <ul>
        <li>模态正交，数学性质稳定，数值表现好</li>
        <li>纯空间统计——模态不含时间动力学信息</li>
        <li>单个模态含多频耦合，物理解释弱</li>
        <li>适合：数据压缩、ROM 基函数构建</li>
      </ul>

      <h2>DMD：动态模态分解</h2>
      <p><strong>核心思想</strong>：假设系统近似线性动力学 x<sub>k+1</sub> = A·x<sub>k</sub>，对时序全场数据做特征分解，提取空间模态 + 对应频率 + 增长/衰减率。</p>

      <h3>特点</h3>
      <ul>
        <li>每个模态对应单一频率，物理可解释性强</li>
        <li>自带增长/衰减系数，可做短期预测</li>
        <li>能判断涡脱落频率、流动稳定性等物理特征</li>
        <li>局强非线性系统精度下降</li>
        <li>适合：流场稳定性分析、全场云图短期预报</li>
      </ul>

      <h2>POD vs DMD 直观对比</h2>
      <table>
        <tr><th>维度</th><th>POD</th><th>DMD</th></tr>
        <tr><td>本质</td><td>空间统计降维</td><td>时空耦合动力学分解</td></tr>
        <tr><td>输入</td><td>全场快照数据</td><td>时序全场数据</td></tr>
        <tr><td>模态含义</td><td>能量最优，含多频耦合</td><td>单频模态，物理可解释</td></tr>
        <tr><td>时间信息</td><td>无</td><td>频率 + 增长/衰减率</td></tr>
        <tr><td>数学基础</td><td>SVD</td><td>线性算子特征分解</td></tr>
        <tr><td>典型应用</td><td>压缩、ROM 基函数</td><td>稳定性分析、短期预测</td></tr>
      </table>

      <h2>DEIM：离散经验插值法</h2>
      <p><strong>定位</strong>：DEIM 不是独立建模算法，是 POD/DMD 降阶框架的配套加速工具。专门解决"降维后非线性项计算依然很慢"的痛点。</p>

      <h3>工作原理</h3>
      <ol>
        <li>从海量网格中自动挑选少数关键插值点</li>
        <li>只用这些关键点的数值插值还原全场非线性项</li>
        <li>将上百万网格计算量压缩到几十点，加速 10~100 倍</li>
      </ol>

      <blockquote>类比：画人像不用铺满像素，只捕捉五官关键点位，插值还原整张脸。</blockquote>

      <h3>特点</h3>
      <ul>
        <li>✅ 极大加速 CAE/流体降阶模型在线计算</li>
        <li>✅ 保证插值精度</li>
        <li>❌ 依附 POD/DMD 存在，不能单独建模预测</li>
        <li>❌ 关键插值点固定，工况偏移后精度下降</li>
      </ul>

      <h2>工程应用建议</h2>
      <table>
        <tr><th>场景</th><th>推荐方法</th></tr>
        <tr><td>全场云图仿真加速</td><td>有限元生成数据 → DMD 提取动态模态 → DEIM 加速非线性项</td></tr>
        <tr><td>传感器时序预测</td><td>NARX 黑箱模型</td></tr>
        <tr><td>复杂多物理场数字孪生</td><td>POD/DMD 降阶 + DEIM 加速 + 神经网络混合模型</td></tr>
      </table>

      <h2>标准流程</h2>
      <p>有限元仿真生成全场数据 → DMD 提取动态模态构建降阶模型 → DEIM 加速非线性求解 → 得到快速仿真模型。若仿真难以建模、只有传感器实测时序数据，则直接用 NARX 搭建黑箱预测模型。</p>
    `
  }
];

/* =============================================
   Homepage Blog Cards
   ============================================= */

function renderHomeBlogCards() {
  const grid = document.getElementById('homeBlogGrid');
  if (!grid) return;

  const latest = blogPosts.slice(0, 3);
  let html = '';
  latest.forEach(post => {
    html += `
      <a href="blog.html#post-${post.id}" class="blog-card" data-id="${post.id}">
        <div class="blog-card-meta">
          <span class="blog-card-date">${post.date}</span>
          <span class="blog-card-cat">${post.category}</span>
        </div>
        <h3>${post.title}</h3>
        <p>${post.excerpt}</p>
      </a>`;
  });
  grid.innerHTML = html;

  // card click → navigate to blog page with post pre-selected
  grid.querySelectorAll('.blog-card').forEach(card => {
    card.addEventListener('click', function(e) {
      const id = this.dataset.id;
      if (id) {
        e.preventDefault();
        window.location.href = `blog.html#post-${id}`;
      }
    });
  });
}

/* =============================================
   Blog Page Functions
   ============================================= */

function buildFileTree() {
  const nav = document.getElementById('fileTreeNav');
  if (!nav) return;

  const cats = {};
  blogPosts.forEach(p => {
    if (!cats[p.category]) cats[p.category] = [];
    cats[p.category].push(p);
  });

  const catList = [
    { name: '调研', icon: '📁' },
    { name: '技术笔记', icon: '📁' },
    { name: '工程实践', icon: '📁' },
    { name: '随笔', icon: '📁' }
  ];

  let html = '';
  catList.forEach(cat => {
    const posts = cats[cat.name];
    if (!posts) return;
    html += `<div class="tree-category">`;
    html += `<div class="tree-cat-label" data-cat="${cat.name}"><span class="arrow">▾</span>${cat.icon} ${cat.name}</div>`;
    html += `<div class="tree-posts">`;
    posts.forEach(p => {
      html += `<a class="tree-post" data-id="${p.id}" href="#post-${p.id}">${p.title}</a>`;
    });
    html += `</div></div>`;
  });

  nav.innerHTML = html;

  nav.querySelectorAll('.tree-cat-label').forEach(lbl => {
    lbl.addEventListener('click', () => {
      lbl.classList.toggle('collapsed');
      const posts = lbl.nextElementSibling;
      if (posts) posts.style.display = lbl.classList.contains('collapsed') ? 'none' : '';
    });
  });

  nav.querySelectorAll('.tree-post').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const id = link.dataset.id;
      const post = blogPosts.find(p => p.id === id);
      if (post) showPost(post, link);
    });
  });
}

function showPost(post, treeEl) {
  document.querySelectorAll('.tree-post').forEach(el => el.classList.remove('active'));
  if (treeEl) treeEl.classList.add('active');

  const main = document.getElementById('blogContent');
  main.innerHTML = `
    <article class="blog-article" id="post-${post.id}">
      <div class="blog-article-header">
        <div class="blog-article-meta">
          <span>${post.date}</span>
          <span class="cat-tag">${post.category}</span>
        </div>
        <h1>${post.title}</h1>
      </div>
      <div class="blog-article-body">${post.content}</div>
    </article>
  `;

  buildTOC();
  setupScrollSpy();
  main.scrollTop = 0;
  window.scrollTo({ top: 60, behavior: 'smooth' });
}

function buildTOC() {
  const tocNav = document.getElementById('tocNav');
  if (!tocNav) return;

  const article = document.querySelector('.blog-article-body');
  if (!article) { tocNav.innerHTML = '<p class="toc-empty">选择文章以查看目录</p>'; return; }

  const headings = article.querySelectorAll('h2, h3');
  if (headings.length === 0) {
    tocNav.innerHTML = '<p class="toc-empty">本文无章节标题</p>';
    return;
  }

  let html = '';
  headings.forEach((h, i) => {
    const tag = h.tagName.toLowerCase();
    const id = `sec-${i}`;
    h.id = id;
    html += `<a class="toc-item toc-${tag}" href="#${id}" data-target="${id}">${h.textContent}</a>`;
  });

  tocNav.innerHTML = html;

  tocNav.querySelectorAll('.toc-item').forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      const target = document.getElementById(item.dataset.target);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

function setupScrollSpy() {
  const tocItems = document.querySelectorAll('.toc-item');
  if (tocItems.length === 0) return;

  const headings = document.querySelectorAll('.blog-article-body h2[id], .blog-article-body h3[id]');
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        tocItems.forEach(item => item.classList.remove('active'));
        const active = document.querySelector(`.toc-item[data-target="${entry.target.id}"]`);
        if (active) active.classList.add('active');
      }
    });
  }, { rootMargin: '-80px 0px -60% 0px', threshold: 0 });

  headings.forEach(h => observer.observe(h));
}

/* =============================================
   Main Page Functions
   ============================================= */

function initMainPage() {
  renderHomeBlogCards();

  // Scroll-triggered fade-in
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) entry.target.classList.add('visible');
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.section-header, .blog-card, .project-card').forEach(el => {
    el.classList.add('fade-in');
    observer.observe(el);
  });

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      if (href === '#' || href === '#projects') {
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
          const offset = 72;
          const pos = target.getBoundingClientRect().top + window.scrollY - offset;
          window.scrollTo({ top: pos, behavior: 'smooth' });
        }
      }
    });
  });
}

/* =============================================
   Init
   ============================================= */

if (document.querySelector('.blog-page')) {
  buildFileTree();

  // Check if URL has #post-xxx hash → auto-open that post
  const hash = window.location.hash;
  if (hash && hash.startsWith('#post-')) {
    const id = hash.replace('#post-', '');
    const post = blogPosts.find(p => p.id === id);
    if (post) {
      const el = document.querySelector(`.tree-post[data-id="${id}"]`);
      showPost(post, el);
    } else {
      // fallback to first post
      if (blogPosts.length > 0) {
        const firstEl = document.querySelector(`.tree-post[data-id="${blogPosts[0].id}"]`);
        showPost(blogPosts[0], firstEl);
      }
    }
  } else if (blogPosts.length > 0) {
    const firstEl = document.querySelector(`.tree-post[data-id="${blogPosts[0].id}"]`);
    showPost(blogPosts[0], firstEl);
  }
} else {
  initMainPage();
}
