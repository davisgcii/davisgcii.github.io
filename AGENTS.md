# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the Astro site source, including pages, components, layouts, and global styles.
- `src/pages/` defines routes (e.g., `src/pages/index.astro`, `src/pages/photos/[slug].astro`).
- `src/content/` stores Markdown content collections (`blog`, `photos`).
- `src/assets/` holds local images referenced by content (e.g., `src/assets/posts/...`, `src/assets/photos/...`).
- `public/` is for static files served as-is.
- There is no dedicated test directory in this repo.

## Build, Test, and Development Commands
- `npm run dev`: start the local Astro dev server.
- `npm run build`: build the production site to `dist/`.
- `npm run preview`: serve the built site locally for production-like testing.

## Coding Style & Naming Conventions
- Language: Astro + TypeScript; use standard `.astro` conventions for templates.
- Indentation: 2 spaces.
- File naming: lowercase with hyphens for content slugs (e.g., `more-efficient-finetuning.md`).
- Asset naming: keep descriptive, short names under each post folder (e.g., `hero.png`, `figure-01.png`).
- Styling: `src/styles/global.css` contains global styles; use existing CSS variables and patterns.

## Testing Guidelines
- No automated test framework is configured.
- Verify changes manually by running `npm run dev` and `npm run preview` for production checks.

## Commit & Pull Request Guidelines
- No formal commit convention is enforced in the repo. Use concise, imperative messages (e.g., `Update photo transitions`).
- For PRs, include a brief summary, list of key changes, and screenshots for UI changes (e.g., photos, blog cards).

## Content & Assets Tips
- Blog posts live in `src/content/blog/`; update frontmatter and asset paths together.
- Photos live in `src/content/photos/`; keep hero images in `src/assets/photos/`.
