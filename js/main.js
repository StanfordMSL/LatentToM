document.addEventListener('DOMContentLoaded', () => {
  const navToggle = document.querySelector('.nav-toggle');
  const navLinks = document.querySelector('.nav-links');

  const closeMenu = () => {
    if (!navToggle || !navLinks) return;
    navToggle.setAttribute('aria-expanded', 'false');
    navLinks.classList.remove('open');
    document.body.classList.remove('menu-open');
  };

  navToggle?.addEventListener('click', () => {
    const isOpen = navToggle.getAttribute('aria-expanded') === 'true';
    navToggle.setAttribute('aria-expanded', String(!isOpen));
    navLinks.classList.toggle('open', !isOpen);
    document.body.classList.toggle('menu-open', !isOpen);
  });

  navLinks?.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeMenu));
  window.addEventListener('resize', () => {
    if (window.innerWidth > 820) closeMenu();
  });

  const sectionLinks = [...document.querySelectorAll('.nav-links a[href^="#"]')];
  const sections = sectionLinks
    .map((link) => document.querySelector(link.getAttribute('href')))
    .filter(Boolean);

  if ('IntersectionObserver' in window && sections.length) {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      sectionLinks.forEach((link) => {
        const active = link.getAttribute('href') === `#${visible.target.id}`;
        link.classList.toggle('active', active);
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }, { rootMargin: '-20% 0px -65%', threshold: [0, 0.15, 0.4] });
    sections.forEach((section) => observer.observe(section));
  }

  const motionReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ambientVideos = [...document.querySelectorAll('video[autoplay]')];
  if (motionReduced) {
    ambientVideos.forEach((video) => video.pause());
  } else if ('IntersectionObserver' in window) {
    const videoObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) entry.target.play().catch(() => {});
        else entry.target.pause();
      });
    }, { rootMargin: '160px 0px', threshold: 0.05 });
    ambientVideos.forEach((video) => videoObserver.observe(video));
  }

  const teamCardToggle = document.querySelector('.team-card-toggle');
  const teamPhotoPanel = document.getElementById('team-photo-panel');
  teamCardToggle?.addEventListener('click', () => {
    if (!teamPhotoPanel) return;
    const willOpen = teamCardToggle.getAttribute('aria-expanded') !== 'true';
    teamCardToggle.setAttribute('aria-expanded', String(willOpen));
    teamPhotoPanel.hidden = !willOpen;
    const hint = teamCardToggle.querySelector('.team-card-toggle-hint');
    const icon = teamCardToggle.querySelector('.team-card-toggle-icon');
    if (hint) hint.textContent = willOpen ? 'Click to hide team photos' : 'Click to reveal team photos';
    if (icon) icon.textContent = willOpen ? '−' : '+';
  });

  document.querySelectorAll('.copy-button[data-copy-target]').forEach((button) => {
    button.addEventListener('click', async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.textContent);
        const original = button.textContent;
        button.textContent = 'Copied';
        window.setTimeout(() => { button.textContent = original; }, 1800);
      } catch {
        button.textContent = 'Select text to copy';
      }
    });
  });
});
