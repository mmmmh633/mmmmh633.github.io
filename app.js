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
      if (post) showArticle(post, id);
    });
  });
}

function renderAllBlogCards() {
  const grid = document.getElementById('allBlogGrid');
  if (!grid) return;

  let html = '';
  blogPosts.forEach(post => {
    html += `
      <div class="blog-card" data-id="${post.id}">
        <div class="blog-card-meta">
          <span class="blog-card-date">${post.date}</span>
          <span class="blog-card-cat">${post.category}</span>
        </div>
        <h3>${post.title}</h3>
        <p>${post.excerpt}</p>
      </div>`;
  });
  grid.innerHTML = html;

  grid.querySelectorAll('.blog-card').forEach(card => {
    card.addEventListener('click', function() {
      const id = this.dataset.id;
      const post = blogPosts.find(p => p.id === id);
      if (post) {
        showArticle(post, id);
        window.history.replaceState(null, '', `#post-${id}`);
      }
    });
  });
}

function showArticle(post, id) {
  // highlight tree item
  document.querySelectorAll('.tree-post').forEach(el => el.classList.remove('active'));
  const treeEl = document.querySelector(`.tree-post[data-id="${id}"]`);
  if (treeEl) treeEl.classList.add('active');

  // switch to article view
  document.getElementById('blogCardsView').style.display = 'none';
  document.getElementById('blogArticleView').style.display = 'block';
  document.getElementById('blogToc').style.display = '';

  const container = document.getElementById('articleContainer');
  container.innerHTML = `
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
  window.scrollTo({ top: 60, behavior: 'smooth' });
}

function showCardsView() {
  document.getElementById('blogArticleView').style.display = 'none';
  document.getElementById('blogCardsView').style.display = '';
  document.getElementById('blogToc').style.display = 'none';
  document.querySelectorAll('.tree-post').forEach(el => el.classList.remove('active'));
  document.getElementById('tocNav').innerHTML = '<p class="toc-empty">选择文章以查看目录</p>';
  window.history.replaceState(null, '', window.location.pathname);
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
}

/* =============================================
   Main Page Functions
   ============================================= */

function initMainPage() {
  renderHomeBlogCards();

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) entry.target.classList.add('visible');
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.section-header, .blog-card, .project-card').forEach(el => {
    el.classList.add('fade-in');
    observer.observe(el);
  });

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
   Share Button
   ============================================= */
function initShareButton() {
  const btn = document.getElementById('shareBtn');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    const url = window.location.href;
    const title = document.title;
    if (navigator.share) {
      try { await navigator.share({ title, url }); } catch {}
    } else {
      try {
        await navigator.clipboard.writeText(url);
        btn.classList.add('copied');
        setTimeout(() => btn.classList.remove('copied'), 1500);
      } catch {
        const input = document.createElement('input');
        input.value = url; document.body.appendChild(input);
        input.select(); document.execCommand('copy');
        document.body.removeChild(input);
        btn.classList.add('copied');
        setTimeout(() => btn.classList.remove('copied'), 1500);
      }
    }
  });
}

/* =============================================
   Init
   ============================================= */

if (document.querySelector('.blog-page')) {
  initShareButton();
  buildFileTree();
  renderAllBlogCards();

  // Back to list button
  document.getElementById('backToList').addEventListener('click', showCardsView);

  // Hide TOC initially (cards view)
  document.getElementById('blogToc').style.display = 'none';

  // If URL has #post-xxx hash → open that article directly
  const hash = window.location.hash;
  if (hash && hash.startsWith('#post-')) {
    const id = hash.replace('#post-', '');
    const post = blogPosts.find(p => p.id === id);
    if (post) showArticle(post, id);
  }
} else {
  initShareButton();
  initMainPage();
}
