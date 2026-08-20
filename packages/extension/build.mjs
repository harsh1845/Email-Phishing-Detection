import * as esbuild from 'esbuild';
import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, 'dist');
const firefox = process.argv.includes('--firefox');

rmSync(dist, { recursive: true, force: true });
mkdirSync(join(dist, 'content'), { recursive: true });
mkdirSync(join(dist, 'icons'), { recursive: true });
mkdirSync(join(dist, 'wasm'), { recursive: true });
mkdirSync(join(dist, 'model'), { recursive: true });
mkdirSync(join(dist, 'assets'), { recursive: true });

const shared = {
  bundle: true,
  target: ['chrome120', 'firefox121'],
  sourcemap: false,
  legalComments: 'none',
  logLevel: 'info',
};

await esbuild.build({
  ...shared,
  entryPoints: {
    background: join(here, 'src/background/service-worker.ts'),
    offscreen: join(here, 'src/offscreen/offscreen.ts'),
    popup: join(here, 'src/popup/popup.ts'),
    interstitial: join(here, 'src/interstitial/interstitial.ts'),
  },
  outdir: dist,
  format: 'esm',
});

await esbuild.build({
  ...shared,
  entryPoints: [join(here, 'src/content/gmail.ts')],
  outfile: join(dist, 'content/gmail.js'),
  format: 'iife',
});

const copy = (from, to) => cpSync(from, to, { recursive: true });
copy(join(here, firefox ? 'src/manifest.firefox.json' : 'src/manifest.chrome.json'), join(dist, 'manifest.json'));
copy(join(here, 'src/content/gmail.css'), join(dist, 'content/gmail.css'));
copy(join(here, 'src/popup/popup.html'), join(dist, 'popup.html'));
copy(join(here, 'src/popup/popup.css'), join(dist, 'popup.css'));
copy(join(here, 'src/interstitial/interstitial.html'), join(dist, 'interstitial.html'));
copy(join(here, 'src/interstitial/interstitial.css'), join(dist, 'interstitial.css'));
copy(join(here, 'src/offscreen/offscreen.html'), join(dist, 'offscreen.html'));
copy(join(here, 'icons'), join(dist, 'icons'));

const assets = join(here, 'src/assets');
if (existsSync(assets)) copy(assets, join(dist, 'assets'));

const model = join(here, 'public/model');
if (existsSync(model)) copy(model, join(dist, 'model'));

const ortWasm = join(here, '../../node_modules/onnxruntime-web/dist');
if (existsSync(ortWasm)) {
  for (const name of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
    const src = join(ortWasm, name);
    if (existsSync(src)) copy(src, join(dist, 'wasm', name));
  }
}

console.log(`Built ${firefox ? 'Firefox' : 'Chrome'} extension → ${dist}`);
