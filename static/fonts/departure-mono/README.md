# Departure Mono

The display font for Reinforced Knowledge headings (h1, h2). Pixel-style monospace.

## Setup (one-time)

1. Visit https://departuremono.com/
2. Click **Download** (free under SIL Open Font License)
3. From the extracted zip, copy `DepartureMono-Regular.woff2` into **this directory** (`static/fonts/departure-mono/`)
4. Commit the woff2 file

The font is referenced from `static/css/base.css` at `/fonts/departure-mono/DepartureMono-Regular.woff2`. Until you drop the file in, h1 and h2 fall back to JetBrains Mono: the site stays functional, it just loses the pixel-style letterform.

## Why self-hosted

Departure Mono is not on Google Fonts. Self-hosting also keeps a third-party request off every pageview.

## License

SIL Open Font License 1.1. Free for commercial and personal use. Full license at https://departuremono.com/license.
