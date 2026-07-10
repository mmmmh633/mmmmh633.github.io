const posts = [];

function render() {
    const el = document.getElementById('blog-posts');
    if (posts.length === 0) {
        el.innerHTML = '<p>博客文章即将发布…</p>';
        return;
    }
    el.innerHTML = posts.map(p => `
        <div class="blog-item">
            <h3>${p.title}</h3>
            <div class="date">${p.date}</div>
            <p>${p.excerpt}</p>
        </div>
    `).join('');
}

render();
