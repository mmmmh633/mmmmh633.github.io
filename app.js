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

if (document.querySelector('.blog-page')) {
  initShareButton();
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
  initShareButton();
  initMainPage();
}
