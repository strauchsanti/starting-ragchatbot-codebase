# Frontend changes: light/dark theme

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

## Light theme CSS variables (second pass, `frontend/style.css`)
- Light palette refined: background `#f1f5f9`, surface `#fff`, border `#cbd5e1`, text `#0f172a`, secondary text `#475569`, primary `#2563eb`/`#1d4ed8` (unchanged, passes AA on light), stronger focus ring.
- New variables, defined for both themes: `--error-*`, `--success-*` (color/bg/border) and `--primary-shadow`. `.error-message`, `.success-message` and the send-button hover shadow now use them instead of hardcoded dark-theme colors.
- Contrast targets (WCAG AA, computed by hand, not tool-verified): primary text on background ~17:1, secondary text on surface ~7.6:1, white on primary ~5.2:1, error/success text on their tints >5:1.
- Bumped `style.css?v=14`; third pass: `style.css?v=16`, `script.js?v=12`.

## JavaScript functionality (third pass)
- `script.js`: `applyTheme()` now adds a temporary `theme-transition` class to `<html>` for 350ms while switching; `style.css` uses it to fade background, border, text, fill, stroke and box-shadow on every element. This replaces the earlier hand-maintained list of transitioned selectors, so nothing is missed (sidebar items, buttons, source links, messages). Hover/typing transitions are unaffected outside the switch.
- The theme also follows live OS `prefers-color-scheme` changes until the user picks one explicitly.

## Implementation audit (fourth pass)
- Theming is driven by CSS custom properties on `:root`, switched via `data-theme` on `<html>` (set pre-paint by an inline script, then by `script.js`).
- Audited remaining hardcoded colors in `style.css`: the welcome-message shadow now uses `--shadow-strong`; `.message-content blockquote` used an undefined `var(--primary)` (border never rendered) and now uses `--primary-color`.
- Added `color-scheme: dark|light` per theme so native scrollbars and form controls match.
- Intentionally unchanged: white text on primary-colored user bubbles/send button (AA on blue in both themes) and the hidden header's gradient.

## Not covered
- The (hidden) header gradient is unchanged.
- Not yet verified in a browser.
