import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    exclude: ['maplibre-gl'],
  },
  build: {
    target: 'es2022',
    // Do not publish source maps: Vercel's immutable static route can reject
    // them with 403s and the client does not need them to run.
    sourcemap: false,
    rollupOptions: {
      output: {
        // Bump the asset namespace whenever a release changes bundling policy.
        // This prevents immutable browser caches from reusing the old bundles
        // that still referenced rejected .js.map files.
        entryFileNames: 'assets/[name]-hf2-[hash].js',
        chunkFileNames: 'assets/[name]-hf2-[hash].js',
        assetFileNames: 'assets/[name]-hf2-[hash][extname]',
        manualChunks(id) {
          if (id.includes('maplibre-gl')) return 'map'
          if (id.includes('/motion/')) return 'motion'
          return undefined
        },
      },
    },
  },
  test: {
    include: ['tests/frontend/**/*.test.{ts,tsx}'],
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/elevation': {
        target: 'https://tiles.mapterhorn.com',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/elevation/, ''),
      },
    },
  },
})
