import { searchPublishedNotes } from './search-index.js';

const themeButton = document.querySelector('#theme-toggle');
themeButton.addEventListener('click', () => {
  const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('curiosite-theme', theme); } catch { /* A private window may disable storage. */ }
  themeButton.setAttribute('aria-label', theme === 'dark' ? '切换为浅色主题' : '切换为深色主题');
});

const dialog = document.querySelector('#search-dialog');
const input = document.querySelector('#search-input');
const form = document.querySelector('#search-form');
const status = document.querySelector('#search-status');
const results = document.querySelector('#search-results');
let controller;
let timer;
let revision = 0;
const staticIndexUrl = document.body.dataset.searchIndex;
let staticIndexPromise;
function loadStaticIndex() {
  if (!staticIndexPromise) {
    staticIndexPromise = fetch(staticIndexUrl).then(async (response) => {
      if (!response.ok) throw new Error('Search index unavailable');
      const index = await response.json();
      if (!Array.isArray(index)) throw new Error('Invalid search index');
      return index;
    }).catch((error) => { staticIndexPromise = undefined; throw error; });
  }
  return staticIndexPromise;
}
function openSearch() { if (!dialog.open) dialog.showModal(); input.focus(); }
document.querySelectorAll('.search-trigger').forEach((button) => button.addEventListener('click', openSearch));
document.querySelector('#search-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => {
  if (event.target !== dialog) return;
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
});
dialog.addEventListener('close', () => { controller?.abort(); revision++; });
document.addEventListener('keydown', (event) => {
  const editing = event.target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName);
  if ((!editing && event.key === '/') || ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k')) {
    event.preventDefault(); openSearch();
  }
});
async function search() {
  clearTimeout(timer);
  controller?.abort();
  const current = ++revision;
  const query = input.value.trim();
  results.replaceChildren();
  if (!query) { status.textContent = '输入关键词，查找笔记中的内容。'; return; }
  status.textContent = '正在查找…';
  controller = new AbortController();
  try {
    let data;
    if (staticIndexUrl) {
      data = { results: searchPublishedNotes(await loadStaticIndex(), query) };
    } else {
      const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`, { signal: controller.signal });
      if (!response.ok) throw new Error('Search failed');
      data = await response.json();
    }
    if (current !== revision) return;
    status.textContent = data.results.length ? `找到 ${data.results.length} 篇笔记` : `没有找到包含「${query}」的公开笔记。`;
    for (const note of data.results) {
      const link = document.createElement('a');
      link.className = 'search-result';
      link.href = `/notes/${encodeURIComponent(note.slug)}${staticIndexUrl ? '/' : ''}`;
      const category = document.createElement('span'); category.className = 'eyebrow'; category.textContent = note.category;
      const title = document.createElement('h3'); title.textContent = note.title;
      const snippet = document.createElement('p'); snippet.textContent = note.snippet;
      link.append(category, title, snippet); results.append(link);
    }
  } catch (error) {
    if (error.name !== 'AbortError' && current === revision) status.textContent = '搜索暂时不可用，请稍后重试。';
  }
}
input.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(search, 180); });
form.addEventListener('submit', (event) => { event.preventDefault(); search(); });

const progress = document.querySelector('.reading-progress');
let waiting = false;
function updateProgress() {
  const height = document.documentElement.scrollHeight - innerHeight;
  progress.value = height > 0 ? Math.min(100, scrollY / height * 100) : 100;
  waiting = false;
}
addEventListener('scroll', () => { if (!waiting) { waiting = true; requestAnimationFrame(updateProgress); } }, { passive: true });
addEventListener('resize', updateProgress);
updateProgress();
const headings = [...document.querySelectorAll('.note-content h2[id], .note-content h3[id], .note-content h4[id]')];
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    const heading = entries.filter((entry) => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (!heading) return;
    document.querySelectorAll('.contents a').forEach((link) => {
      const active = decodeURIComponent(link.hash.slice(1)) === heading.target.id;
      link.classList.toggle('current', active);
      if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
    });
  }, { rootMargin: '-95px 0px -65% 0px', threshold: 0 });
  headings.forEach((heading) => observer.observe(heading));
}
document.querySelectorAll('.mobile-contents a').forEach((link) => link.addEventListener('click', () => {
  document.querySelector('.mobile-contents').open = false;
}));
