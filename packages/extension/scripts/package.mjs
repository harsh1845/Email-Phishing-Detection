import { existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const extRoot = join(here, '..');
const target = process.argv[2] === 'firefox' ? 'firefox' : 'chrome';
const dist = join(extRoot, 'dist');
if (!existsSync(dist)) {
  console.error('Run the build first');
  process.exit(1);
}
const outDir = join(extRoot, '../../dist');
mkdirSync(outDir, { recursive: true });
const zip = join(outDir, `phishguard-${target}.zip`);
const result = spawnSync('zip', ['-r', '-q', zip, '.'], { cwd: dist, stdio: 'inherit' });
if (result.status === 0) console.log('Wrote', zip);
process.exit(result.status ?? 1);
