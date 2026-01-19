# Personal Site Plan (Astro + TypeScript)

## Assumptions
- Use Astro with TypeScript and a static output suitable for GitHub Pages.
- Start with minimal content sections: home/intro, blog index, photos grid, and about/contact.
- Content managed in Markdown initially (MDX can be added later if needed).
- Deploy target: GitHub Pages.

## Goals
- Build a clean, personal site inspired by mckerlie.com with strong typography and generous whitespace.
- Support blog posts and photo gallery with tags.
- Fast, accessible, and easy to maintain.

## Work Plan
1. Project setup: initialize Astro + TypeScript, basic configuration, and linting/formatting.
2. Information architecture: define routes (/, /blog, /blog/[slug], /photos, /about) and content collections.
3. Design system: typography, color palette, layout grid, and reusable components.
4. Core pages: implement home, blog list/detail, photos grid/detail, and about/contact.
5. Content: add sample posts/photos and templates for future entries.
6. Deployment: configure GitHub Pages and CI workflow; document deploy steps.
7. Polish: SEO, RSS, sitemap, image optimization, and performance checks.

## Decisions Locked In
- Site name: George Davis.
- Typography: monospaced.
- Blog content format: Markdown.
- Photos: local images in the repo.
- Hosting: GitHub Pages.

## Open Questions
- Do you want a short tagline or intro blurb on the homepage?
- Any preferred accent color(s) to pair with the mono typography?
