# Frontend changes: light/dark theme toggle

## `frontend/index.html`
- Added a `#themeToggle` button (fixed, top-right) containing inline moon and sun SVG icons. Both are `aria-hidden`; the button carries `aria-label` and `aria-pressed`.
- Added an inline script in `<head>` that applies the saved theme (or the OS `prefers-color-scheme`) to `<html data-theme>` before first paint, to avoid a flash.
- Bumped cache-busting versions: `style.css?v=13`, `script.js?v=11`.

## `frontend/style.css`
- Added a `:root[data-theme="light"]` block overriding the color variables (background, surface, text, border, assistant message, shadow, welcome background).
- Added a `--code-bg` variable; `.message-content code` and `pre` now use it instead of a hardcoded dark overlay.
- Added `.theme-toggle` styles: 40px round button, hover scale, `:focus-visible` ring, and rotate/fade transitions between the moon and sun icons.
- Added 0.3s color transitions on the main surfaces, and a `prefers-reduced-motion` rule that effectively disables transitions.

## `frontend/script.js`
- Added `setupThemeToggle()`, called on `DOMContentLoaded`. It toggles `data-theme` on `<html>`, saves the choice to `localStorage` (wrapped in try/catch), and keeps `aria-label` and `aria-pressed` in sync.

## Not covered
- Hardcoded colors in the error/success banners and the header gradient were not adjusted for light mode.
- Not yet verified in a browser.
