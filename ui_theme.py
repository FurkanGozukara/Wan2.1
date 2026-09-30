"""Shared UI styling: theme, per-button colours, page CSS and client-side helpers.

The look follows the SECourses IndexTTS Premium and Whisper-WebUI apps: the stock
Gradio ``Origin`` theme, dark by default with an instant light/dark switch, and
every action button in its own hue so each tab has its own colours instead of a
wall of identical controls. Everything except the button palette is written
against the theme's CSS variables, so light and dark mode stay in sync.
"""

import gradio as gr

__all__ = ["BUTTON_COLORS", "CSS", "HEAD", "TOGGLE_SECTIONS_JS", "TOGGLE_THEME_JS", "app_theme", "btn"]


def app_theme(name=None):
    """Return the stock Origin theme, or the Gradio theme named on the command line."""
    if not name or str(name).strip().lower() == "origin":
        return gr.themes.Origin()
    return name


# Every action button gets its own hue as a (deep, mid, bright) triple. The gradient
# runs deep -> mid -> bright so the face has depth, the glow is built from the mid
# tone, and white text stays readable on all three stops.
BUTTON_HUES = {
    "emerald": ("#065f46", "#059669", "#34d399"),
    "green":   ("#166534", "#16a34a", "#4ade80"),
    "lime":    ("#3f6212", "#65a30d", "#a3e635"),
    "teal":    ("#115e59", "#0d9488", "#2dd4bf"),
    "cyan":    ("#155e75", "#0891b2", "#22d3ee"),
    "sky":     ("#075985", "#0284c7", "#38bdf8"),
    "blue":    ("#1e40af", "#2563eb", "#60a5fa"),
    "indigo":  ("#3730a3", "#4f46e5", "#818cf8"),
    "violet":  ("#5b21b6", "#7c3aed", "#a78bfa"),
    "purple":  ("#6b21a8", "#9333ea", "#c084fc"),
    "fuchsia": ("#86198f", "#c026d3", "#e879f9"),
    "pink":    ("#9d174d", "#db2777", "#f9a8d4"),
    "rose":    ("#9f1239", "#e11d48", "#fda4af"),
    "red":     ("#991b1b", "#dc2626", "#f87171"),
    "crimson": ("#7f1d1d", "#b91c1c", "#ef4444"),
    "orange":  ("#9a3412", "#ea580c", "#fb923c"),
    "amber":   ("#92400e", "#d97706", "#fbbf24"),
    "gold":    ("#854d0e", "#ca8a04", "#fde047"),
    "bronze":  ("#5c3a21", "#8b5a2b", "#d4a373"),
    "coral":   ("#9a3b2e", "#e0573e", "#ffa08a"),
    "slate":   ("#334155", "#475569", "#94a3b8"),
    "gray":    ("#3f3f46", "#52525b", "#a1a1aa"),
}
BUTTON_COLORS = tuple(BUTTON_HUES)


def btn(color, *extra):
    """``elem_classes`` for a coloured action button: same height and type size, only the hue changes."""
    if color not in BUTTON_HUES:
        raise ValueError(f"Unknown button colour: {color}")
    return ["ax", f"ax-{color}", *extra]


def _rgb(value):
    digits = value.lstrip("#")
    return tuple(int(digits[index: index + 2], 16) for index in (0, 2, 4))


def _button_palette_css():
    rules = []
    for name, (deep, mid, bright) in BUTTON_HUES.items():
        red, green, blue = _rgb(mid)
        bright_rgb = ", ".join(str(part) for part in _rgb(bright))
        rules.append(
            f"""
button.ax-{name} {{
  background: linear-gradient(135deg, {deep} 0%, {mid} 55%, {bright} 100%) !important;
  border-color: rgba({bright_rgb}, .72) !important;
  box-shadow: 0 8px 20px rgba({red}, {green}, {blue}, .30), inset 0 1px 0 rgba(255, 255, 255, .20) !important;
}}
button.ax-{name}:hover:not(:disabled) {{
  border-color: rgba({bright_rgb}, .98) !important;
  box-shadow: 0 12px 26px rgba({red}, {green}, {blue}, .44), inset 0 1px 0 rgba(255, 255, 255, .28) !important;
}}
body:not(.dark) button.ax-{name} {{
  background: linear-gradient(135deg, {deep} 0%, {mid} 66%, {bright} 100%) !important;
  border-color: {mid} !important;
  box-shadow: 0 7px 17px rgba({red}, {green}, {blue}, .26), inset 0 1px 0 rgba(255, 255, 255, .26) !important;
}}
body:not(.dark) button.ax-{name}:hover:not(:disabled) {{
  box-shadow: 0 11px 24px rgba({red}, {green}, {blue}, .38), inset 0 1px 0 rgba(255, 255, 255, .32) !important;
}}"""
        )
    return "\n".join(rules)


_BASE_CSS = r"""
:root {
  --ax-ok: #047857;
  --ax-warn: #b45309;
  --ax-error: #be123c;
  --ax-accent: #0f766e;
}
:root.dark, :root .dark {
  --ax-ok: #10b981;
  --ax-warn: #f59e0b;
  --ax-error: #fb7185;
  --ax-accent: #14b8a6;
}

/* Action buttons: one height, weight and type size so rows of controls share a
   baseline; only the hue changes. Motion is limited to hover and press, nothing
   animates while the page is idle, so long generations stay smooth. */
button.ax {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--size-2);
  min-height: 44px;
  padding: var(--size-2) var(--size-4) !important;
  border-width: 1px !important;
  border-style: solid !important;
  border-radius: var(--radius-lg) !important;
  color: #f8fafc !important;
  font-size: var(--text-md) !important;
  font-weight: 650 !important;
  line-height: 1.25 !important;
  text-align: center;
  text-shadow: 0 1px 2px rgba(2, 6, 23, .45) !important;
  transition: transform 140ms ease, filter 140ms ease, box-shadow 140ms ease;
}
button.ax:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.06); }
button.ax:active:not(:disabled) { transform: translateY(1px); filter: brightness(.96); }
button.ax:focus-visible { outline: 2px solid #bae6fd; outline-offset: 2px; }
button.ax:disabled { filter: grayscale(.45) opacity(.62); transform: none; box-shadow: none !important; cursor: not-allowed; }
button.ax.ax-lg { min-height: 54px; font-size: var(--text-lg) !important; letter-spacing: .02em; }
.row > button.ax { align-self: center; }

/* Header */
.app-header {
  align-items: center;
  flex-wrap: nowrap;
  gap: var(--size-4);
  padding-bottom: var(--size-3);
  border-bottom: 1px solid var(--border-color-primary);
}
.app-header > :first-child { flex: 1 1 auto; min-width: 0; }
.app-header h1 { margin: 0 !important; line-height: 1.2; }
.app-header p { margin: var(--size-1) 0 0 !important; color: var(--body-text-color-subdued); }
.app-header a { text-decoration: none; }
.app-header a:hover { text-decoration: underline; }
.row.header-actions {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
  flex-wrap: nowrap;
  justify-content: flex-end;
  gap: var(--size-2);
}
.header-actions button.ax { flex: 0 0 auto; white-space: nowrap; }

/* Config bar and small helpers */
.row.preset-bar { align-items: flex-end; }
.row.preset-bar > button.ax { align-self: flex-end !important; margin-bottom: 2px; }
.row.input-action-row { align-items: flex-end; }
.row.input-action-row > button.ax { align-self: flex-end !important; margin-bottom: 11px; }
.preset-status { min-height: var(--size-6); font-size: var(--text-sm); color: var(--body-text-color-subdued); }
.side-actions { gap: var(--size-3) !important; }
.column.group-panel { padding: var(--block-padding) !important; background: var(--block-background-fill); }
.column.group-panel .prose { color: var(--body-text-color); }
.section-note { font-size: var(--text-sm); color: var(--body-text-color-subdued); }
.section-note p { margin: 0 !important; }

/* Status cards (models tab) */
.model-status-table table { width: 100%; border-collapse: collapse; }
.model-status-table th, .model-status-table td { padding: var(--size-2); border: 1px solid var(--border-color-primary); text-align: left; }
.model-status-table th { background: var(--background-fill-secondary); }
.model-status-table .ok { color: var(--ax-ok); font-weight: 700; }
.model-status-table .missing { color: var(--ax-error); font-weight: 700; }
.vram-table table { width: 100%; border-collapse: collapse; }
.vram-table th, .vram-table td { padding: var(--size-1) var(--size-2); border: 1px solid var(--border-color-primary); text-align: left; }
.vram-table th { background: var(--background-fill-secondary); }

/* Log boxes: fixed height, scroll inside */
.log-box textarea { font-family: var(--font-mono); font-size: var(--text-sm) !important; }

@media (max-width: 900px) {
  .app-header { flex-wrap: wrap; }
  .row.header-actions { flex: 1 1 100% !important; width: 100% !important; flex-wrap: wrap; justify-content: flex-start; }
  .header-actions button.ax { flex: 1 1 180px; width: auto !important; min-width: 0 !important; white-space: normal; }
}

@media (prefers-reduced-motion: reduce) {
  button.ax { transition: none; }
  button.ax:hover:not(:disabled), button.ax:active:not(:disabled) { transform: none; }
}
"""

CSS = _BASE_CSS + _button_palette_css() + "\n"


# Dark is the default the first time the app is opened. The choice is stored in
# localStorage and mirrored into the ``__theme`` query parameter so a reload or a
# bookmark restores it before Gradio paints the page.
HEAD = """
<meta name="color-scheme" content="dark light">
<script>
(function () {
  var KEY = "secourses.wan21.theme";
  function resolve() {
    var url = new URL(window.location.href);
    var param = url.searchParams.get("__theme");
    var stored = null;
    try { stored = window.localStorage.getItem(KEY); } catch (e) {}
    var mode = param || stored || "dark";
    if (mode !== "light") { mode = "dark"; }
    try { window.localStorage.setItem(KEY, mode); } catch (e) {}
    if (param !== mode) {
      url.searchParams.set("__theme", mode);
      window.history.replaceState(null, "", url.toString());
    }
    return mode;
  }
  function paint(mode) {
    if (!document.body) { return false; }
    document.body.classList.toggle("dark", mode === "dark");
    return true;
  }
  var mode = resolve();
  if (!paint(mode)) {
    document.addEventListener("DOMContentLoaded", function () { paint(mode); });
  }
})();
</script>
"""


# Switching themes only swaps the ``dark`` class Gradio keys off, so it is instant:
# no reload and no round trip to the server, even while a generation holds the queue.
TOGGLE_THEME_JS = """
() => {
  const dark = !document.body.classList.contains("dark");
  document.body.classList.toggle("dark", dark);
  const mode = dark ? "dark" : "light";
  try { window.localStorage.setItem("secourses.wan21.theme", mode); } catch (e) {}
  const url = new URL(window.location.href);
  url.searchParams.set("__theme", mode);
  window.history.replaceState(null, "", url.toString());
}
"""


# Expand or collapse every accordion on the tab that is on screen.
TOGGLE_SECTIONS_JS = """
async () => {
  const visible = (element) =>
    Boolean(element && (element.offsetWidth || element.offsetHeight || element.getClientRects().length));
  const tabs = document.querySelector("#main-tabs");
  const panel = tabs
    ? Array.from(tabs.querySelectorAll(":scope > .tabitem")).find(visible)
    : null;
  const scope = panel || document;
  const heads = () => Array.from(scope.querySelectorAll("button.label-wrap")).filter(visible);
  const first = heads();
  if (!first.length) { return; }
  const expand = first.some((head) => !head.classList.contains("open"));
  for (let pass = 0; pass < 6; pass++) {
    const pending = heads().filter((head) => head.classList.contains("open") !== expand);
    if (!pending.length) { break; }
    pending.forEach((head) => head.click());
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  }
}
"""
