document.addEventListener('DOMContentLoaded', function () {
  var body = document.querySelector('div[itemprop="articleBody"]');
  if (!body) return;
  var heads = body.querySelectorAll('h2, h3');
  if (heads.length < 2) return;

  var nav = document.createElement('nav');
  nav.id = 'page-toc';
  var title = document.createElement('div');
  title.className = 'page-toc-title';
  title.textContent = 'On this page';
  nav.appendChild(title);

  var ul = document.createElement('ul');
  var links = [];
  heads.forEach(function (h) {
    var sec = h.closest('section, div.section');
    var id = (sec && sec.id) || h.id;
    if (!id) return;
    var li = document.createElement('li');
    li.className = 'toc-' + h.tagName.toLowerCase();
    var a = document.createElement('a');
    a.href = '#' + id;
    a.textContent = h.textContent.replace('\u00b6', '').replace('\uf0c1', '').trim();
    li.appendChild(a);
    ul.appendChild(li);
    links.push({ a: a, el: h });
  });
  nav.appendChild(ul);
  document.body.appendChild(nav);

  function onScroll() {
    var cur = links[0];
    links.forEach(function (l) {
      if (l.el.getBoundingClientRect().top < 120) cur = l;
    });
    links.forEach(function (l) { l.a.classList.toggle('active', l === cur); });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
});
