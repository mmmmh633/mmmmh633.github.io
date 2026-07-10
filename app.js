/* =============================================
   Blog Posts Data
   ============================================= */
const blogPosts = [
  {
    id: 'pinn-intro',
    title: 'PINN：物理信息神经网络入门',
    date: '2026-06-20',
    category: '技术笔记',
    content: `
      <p>物理信息神经网络（Physics-Informed Neural Networks, PINN）将物理定律以偏微分方程残差的形式嵌入神经网络损失函数，使模型在数据稀疏时仍能收敛到符合物理约束的解。</p>

      <h2>核心思想</h2>
      <p>传统神经网络仅依赖数据驱动，而 PINN 在损失函数中额外引入<strong>物理残差项</strong>：</p>
      <pre><code>Loss = Loss_data + λ * Loss_pde</code></pre>
      <p>其中 <code>Loss_pde</code> 是偏微分方程在配点上的残差。通过自动微分计算各阶导数，无需手动推导。</p>

      <h3>正向问题</h3>
      <p>已知边界条件和物理参数，求解场分布。PINN 直接将坐标作为输入，场量作为输出，通过 PDE 残差约束训练。</p>

      <h3>逆向问题</h3>
      <p>已知部分观测数据，反推物理参数。PINN 将未知参数设为可训练变量，与网络权重同步优化。</p>

      <h2>技术要点</h2>
      <ul>
        <li><strong>自动微分</strong>：框架内置 AD，精确计算梯度，无需数值差分</li>
        <li><strong>配点采样</strong>：在域内和边界上采样配点，残差在此处计算</li>
        <li><strong>多任务优化</strong>：数据项与 PDE 项梯度尺度不均衡，需动态调节权重</li>
        <li><strong>硬约束 vs 软约束</strong>：可将边界条件硬编码进网络结构，提升收敛速度</li>
      </ul>

      <h2>常用框架</h2>
      <p>目前主流工具有 PaddleScience、NVIDIA Modulus（原 SimNet）、DeepXDE 等。PaddleScience 基于 PaddlePaddle，对国产硬件适配较好。</p>

      <blockquote>PINN 的优势在于小数据与强物理约束场景。当数据充足时，纯数据驱动方法通常更高效。</blockquote>
    `
  },
  {
    id: 'paddlescience-guide',
    title: 'PaddleScience 实践指南',
    date: '2026-06-25',
    category: '技术笔记',
    content: `
      <p>PaddleScience 是百度基于 PaddlePaddle 推出的 AI for Science 工具包，面向流体力学、固体力学、电磁学等领域的仿真加速。</p>

      <h2>架构概览</h2>
      <p>核心模块包括：几何定义（Geometry）、方程约束（Equation）、神经网络（Model）、优化器与调度器，以及后处理与可视化。各模块通过配置文件串联，降低使用门槛。</p>

      <h2>几何与网格</h2>
      <p>支持矩形、圆形、L 形等基础几何体，以及布尔组合。内部自动生成采样点，也可自定义采样策略。</p>
      <pre><code>from ppsci.geometry import Rectangle, PointCloud
geom = Rectangle((-1,-1), (1,1))</code></pre>

      <h2>方程定义</h2>
      <p>通过符号化或继承基类定义 PDE。以内置 Navier-Stokes 为例，框架自动处理无量纲化和导数计算。</p>

      <h3>自定义方程</h3>
      <p>继承 <code>equation.PDE</code> 基类，重写 <code>pde</code> 方法，返回各残差项的字典。自动微分计算所有必要偏导数。</p>

      <h2>训练与调优</h2>
      <ul>
        <li>推荐使用 Adam + L-BFGS 两阶段训练策略</li>
        <li>损失权重可动态调整（如 NTK、GradNorm）</li>
        <li>学习率预热 + 余弦退火通常表现稳定</li>
      </ul>

      <h2>与 PhysicsNemo 的对比</h2>
      <p>PhysicsNemo（NVIDIA Modulus）生态更成熟，支持多 GPU 并行和符号式方程定义。PaddleScience 的优势在于国产化适配和中文社区支持。</p>
    `
  },
  {
    id: 'transient-dt',
    title: '数字孪生中的瞬态分析方法',
    date: '2026-07-02',
    category: '工程实践',
    content: `
      <p>数字孪生（Digital Twin）要求虚拟模型与物理实体实时同步。对于动态系统，这意味着需要高效的瞬态分析方法来捕获时变行为。</p>

      <h2>瞬态 vs 稳态</h2>
      <p>稳态分析求解系统平衡状态，计算量小但对动态过程无能为力。瞬态分析以时间步进方式求解，能捕获启动、关停、突变等关键过程，但计算成本成倍增加。</p>

      <h2>加速策略</h2>
      <h3>降阶模型（ROM）</h3>
      <p>通过 POD、DMD 等方法将高维系统投影到低维子空间，保留主要动力学特征。配合 Galerkin 投影，可将求解时间从小时级压缩到秒级。</p>

      <h3>代理模型</h3>
      <p>用神经网络学习「输入参数 → 瞬态响应」的映射，替代传统求解器。PINN 和 DeepONet 在此方向表现突出。</p>

      <h3>自适应时间步长</h3>
      <p>根据局部误差估计动态调整步长，在变化剧烈处加密、平稳处放宽，平衡精度与效率。</p>

      <h2>实际挑战</h2>
      <ul>
        <li><strong>多尺度耦合</strong>：不同物理场的时间尺度差异大（如热 vs 结构），需多速率积分策略</li>
        <li><strong>实时性要求</strong>：数字孪生要求延迟在毫秒级，传统求解器难以满足</li>
        <li><strong>不确定性量化</strong>：传感器噪声和模型误差的传播需要在瞬态框架下分析</li>
      </ul>

      <blockquote>瞬态分析是数字孪生从「事后回放」走向「实时同步」的核心技术瓶颈。</blockquote>
    `
  },
  {
    id: 'gpu-simulation',
    title: 'GPU 加速科学计算实践',
    date: '2026-07-05',
    category: '工程实践',
    content: `
      <p>科学计算从 CPU 向 GPU 迁移是大势所趋。本文记录 GPU 加速在仿真场景中的关键技术和踩坑经验。</p>

      <h2>硬件选型</h2>
      <p>双精度（FP64）性能是科学计算的核心指标。消费级 GPU（如 RTX 系列）双精度被大幅阉割，推荐使用 Tesla/Quadro 系列或云 GPU 实例。</p>

      <h2>并行模式</h2>
      <h3>数据并行</h3>
      <p>多 GPU 各持完整模型副本，并行处理不同数据批次。适合参数扫描和蒙特卡洛类任务，实现简单但对大模型不友好。</p>

      <h3>模型并行</h3>
      <p>模型拆分到多 GPU，适合超大网络。在 PINN 场景中，可按物理区域划分子网络，各自负责不同子域。</p>

      <h2>CUDA 编程要点</h2>
      <ul>
        <li>合并内存访问：确保相邻线程访问相邻地址</li>
        <li>共享内存利用：减少全局内存读写，适合模板计算类 kernel</li>
        <li>异步传输：使用 CUDA Stream 重叠计算与数据传输</li>
      </ul>

      <h2>框架选择</h2>
      <p>PyTorch/PaddlePaddle 提供了良好的 GPU 抽象。对于自定义算子，推荐用 CUDA C++ 编写扩展，通过 pybind11 或 torch 的 C++ API 接入。</p>

      <p>在数字孪生场景中，GPU 加速使实时瞬态仿真成为可能。配合降阶模型，可将传统需要数小时的仿真压缩到亚秒级。</p>
    `
  },
  {
    id: 'mechanics-to-ai',
    title: '从力学到 AI：一个工程师的转型之路',
    date: '2026-07-08',
    category: '随笔',
    content: `
      <p>本科四年 + 硕士三年，我在力学系度过了七年。如今站在仿真与 AI 的交汇处，回看这段路径，想写下来给同样在转型中的人一些参考。</p>

      <h2>力学给了我什么</h2>
      <p>力学的核心训练是<strong>建模思维</strong>：把一个物理现象抽象成偏微分方程，识别主导项，做量纲分析，判断边界条件。这套思维在 AI 领域同样适用——理解问题结构比调参更重要。</p>

      <p>计算力学让我接触了数值方法：有限元、有限差分、谱方法。这些离散化技术是 PINN 的对比基准，不理解传统方法就无法判断新方法的优劣。</p>

      <h2>为什么要转向 AI</h2>
      <p>传统仿真有两个瓶颈：一是速度——复杂模型的瞬态分析动辄数小时；二是泛化——每次改参数都要重新求解。AI 方法（特别是 PINN 和代理模型）正好能缓解这两个问题。</p>

      <h2>学习路径</h2>
      <ul>
        <li><strong>第一阶段</strong>：补 Python 和基本 ML（3 个月）。吴恩达课程 + 李沐《动手学深度学习》</li>
        <li><strong>第二阶段</strong>：深入 PINN 文献（2 个月）。从 Raissi 的原始论文开始，跟踪到最新进展</li>
        <li><strong>第三阶段</strong>：动手复现（3 个月）。用 PaddleScience 复现基准案例，理解工程细节</li>
      </ul>

      <h2>心得</h2>
      <p>最大的体会是：<strong>不要丢掉工程背景</strong>。纯 AI 出身的人懂网络结构但不懂物理；纯力学出身的人懂方程但不懂优化。站在中间是最稀缺的位置。</p>

      <blockquote>转型不是抛弃过去，而是给已有的知识体系装上新的引擎。</blockquote>
    `
  }
];

/* =============================================
   Blog Page Functions
   ============================================= */

function buildFileTree() {
  const nav = document.getElementById('fileTreeNav');
  if (!nav) return;

  // group posts by category
  const cats = {};
  blogPosts.forEach(p => {
    if (!cats[p.category]) cats[p.category] = [];
    cats[p.category].push(p);
  });

  const catList = [
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

  // category collapse
  nav.querySelectorAll('.tree-cat-label').forEach(lbl => {
    lbl.addEventListener('click', () => {
      lbl.classList.toggle('collapsed');
      const posts = lbl.nextElementSibling;
      if (posts) posts.style.display = lbl.classList.contains('collapsed') ? 'none' : '';
    });
  });

  // post click
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
  // update tree active state
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

  // scroll to top of content
  main.scrollTop = 0;
  window.scrollTo({ top: 56, behavior: 'smooth' });
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

  // TOC click
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
  // Scroll-triggered fade-in
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) entry.target.classList.add('visible');
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.section-header, .blog-card, .project-card, .highlight, .contact-link').forEach(el => {
    el.classList.add('fade-in');
    observer.observe(el);
  });

  // Smooth scroll offset for fixed nav (anchor links on index.html)
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      if (href === '#') return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        const offset = 72;
        const pos = target.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({ top: pos, behavior: 'smooth' });
      }
    });
  });
}

/* =============================================
   Init
   ============================================= */

if (document.querySelector('.blog-page')) {
  // blog page
  buildFileTree();
  // auto-open first post
  if (blogPosts.length > 0) {
    const firstEl = document.querySelector(`.tree-post[data-id="${blogPosts[0].id}"]`);
    showPost(blogPosts[0], firstEl);
  }
} else {
  // main page
  initMainPage();
}
