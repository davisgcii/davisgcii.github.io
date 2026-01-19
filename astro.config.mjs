// @ts-check

import sitemap from '@astrojs/sitemap';
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
	site: 'https://davisgcii.github.io',
	integrations: [sitemap()],
	markdown: {
		syntaxHighlight: 'shiki',
		shikiConfig: {
			themes: {
				light: 'catppuccin-latte',
				dark: 'catppuccin-mocha',
			},
			defaultColor: false,
		},
	},
});
