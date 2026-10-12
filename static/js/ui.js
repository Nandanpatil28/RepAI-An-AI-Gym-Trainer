(() => {
  'use strict';

  const root = document.documentElement;
  const themeToggle = document.querySelector('.theme-toggle');
  const themeIcon = document.querySelector('.theme-toggle-icon');
  const themeText = document.querySelector('.theme-toggle-text');
  const menuToggle = document.querySelector('.menu-toggle');
  const sidebar = document.querySelector('.app-sidebar');
  const overlay = document.querySelector('.sidebar-overlay');
  const closeButtons = document.querySelectorAll('[data-sidebar-close]');
  const sidebarLinks = document.querySelectorAll('.sidebar-nav a');

  function setTheme(theme) {
    const nextTheme = theme === 'light' ? 'light' : 'dark';
    root.dataset.theme = nextTheme;
    try {
      localStorage.setItem('repai-theme', nextTheme);
    } catch (error) {
      // Theme still works for the current page when storage is unavailable.
    }
    if (themeToggle) {
      const isLight = nextTheme === 'light';
      themeToggle.setAttribute('aria-pressed', isLight ? 'true' : 'false');
      themeToggle.setAttribute('aria-label', isLight ? 'Switch to dark mode' : 'Switch to light mode');
    }
    if (themeIcon) themeIcon.textContent = nextTheme === 'light' ? '\u2600' : '\u263e';
    if (themeText) themeText.textContent = nextTheme === 'light' ? 'Light' : 'Dark';
  }

  function setSidebarOpen(open) {
    if (!sidebar || !overlay || !menuToggle) return;
    document.body.classList.toggle('sidebar-open', open);
    sidebar.setAttribute('aria-hidden', open ? 'false' : 'true');
    menuToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    menuToggle.setAttribute('aria-label', open ? 'Close navigation menu' : 'Open navigation menu');
    overlay.hidden = !open;
    if (open) {
      const activeLink = sidebar.querySelector('a.active') || sidebar.querySelector('a');
      window.setTimeout(() => activeLink && activeLink.focus({ preventScroll: true }), 160);
    } else {
      menuToggle.focus({ preventScroll: true });
    }
  }

  setTheme(root.dataset.theme);

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      setTheme(root.dataset.theme === 'light' ? 'dark' : 'light');
    });
  }

  if (menuToggle) {
    menuToggle.addEventListener('click', () => {
      setSidebarOpen(!document.body.classList.contains('sidebar-open'));
    });
  }

  closeButtons.forEach(button => {
    button.addEventListener('click', () => setSidebarOpen(false));
  });

  sidebarLinks.forEach(link => {
    link.addEventListener('click', () => setSidebarOpen(false));
  });

  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && document.body.classList.contains('sidebar-open')) {
      setSidebarOpen(false);
    }
  });
})();
