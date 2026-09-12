import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'

const version = 'hf3'
const sourceDirectory = new URL('../node_modules/maplibre-gl/dist/', import.meta.url)
const outputDirectory = new URL('../dist/assets/', import.meta.url)
const sharedFileName = `maplibre-gl-shared-${version}.mjs`
const workerFileName = `maplibre-gl-worker-${version}.mjs`

await mkdir(outputDirectory, { recursive: true })
await copyFile(new URL('maplibre-gl-shared.mjs', sourceDirectory), new URL(sharedFileName, outputDirectory))

const workerSource = await readFile(new URL('maplibre-gl-worker.mjs', sourceDirectory), 'utf8')
const versionedWorkerSource = workerSource.replace(
  './maplibre-gl-shared.mjs',
  `./${sharedFileName}`,
)

if (versionedWorkerSource === workerSource) {
  throw new Error('MapLibre worker no longer imports the expected shared module')
}

await writeFile(new URL(workerFileName, outputDirectory), versionedWorkerSource)
