const searchInput = document.getElementById('searchInput');
const sections = Array.from(document.querySelectorAll('.searchable-section'));
const tocLinks = Array.from(document.querySelectorAll('.toc a'));
const backToTop = document.getElementById('backToTop');

searchInput.addEventListener('input', function () {
  const query = this.value.trim().toLowerCase();

  sections.forEach(section => {
    const text = (section.innerText + ' ' + (section.dataset.search || '')).toLowerCase();
    const matches = !query || text.includes(query);
    section.classList.toggle('hidden-by-search', !matches);
  });
});

function updateActiveTOC() {
  let currentId = '';

  sections.forEach(section => {
    const rect = section.getBoundingClientRect();
    if (rect.top <= 140 && rect.bottom >= 140) {
      currentId = section.id;
    }
  });

  tocLinks.forEach(link => {
    link.classList.toggle('active', link.getAttribute('href') === '#' + currentId);
  });
}

window.addEventListener('scroll', function () {
  updateActiveTOC();

  if (window.scrollY > 400) {
    backToTop.classList.add('show');
  } else {
    backToTop.classList.remove('show');
  }
});

backToTop.addEventListener('click', function () {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

updateActiveTOC();
