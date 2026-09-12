// Capture hackathon media: screenshots and a UI walkthrough video.
//
// Usage: node scripts/capture_media.mjs [base-url]
// Defaults to the production site. Writes HF/media/screenshots/*.png and
// records HF/media/hf-walkthrough.webm (convert to mp4 with ffmpeg).

import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const BASE = process.argv[2] ?? 'https://humanitarian-forecast.vercel.app'
const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const SHOTS = path.join(ROOT, 'media', 'screenshots')
mkdirSync(SHOTS, { recursive: true })

const closeTutorial = async (page) => {
  const close = page.getByRole('button', { name: 'Close tutorial' })
  if (await close.count()) await close.click().catch(() => {})
}

async function heroShots() {
  const browser = await chromium.launch()
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } })
  const page = await context.newPage()

  // Tutorial card on first visit, map flight running behind it.
  await page.goto(BASE + '/', { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(2500)
  await page.screenshot({ path: path.join(SHOTS, '01-tutorial.png') })

  // Walk through the tutorial steps so the video-style capture matches real use.
  for (let step = 0; step < 5; step++) {
    await page.getByRole('button', { name: step === 4 ? 'Enter HF' : 'Next' }).click()
    await page.waitForTimeout(600)
  }
  await page.waitForTimeout(3200) // let the entry flight land and terrain settle
  await page.screenshot({ path: path.join(SHOTS, '02-globe-forecast.png') })

  // Model registry with the live retrained model and its history.
  await page.getByRole('button', { name: /forecast model: LIVE/ }).click()
  await page.waitForTimeout(1200)
  await page.screenshot({ path: path.join(SHOTS, '03-model-registry.png') })
  await page.getByRole('button', { name: 'Close model registry' }).click()

  // Data table and the zone inspector behind a selected row.
  await page.getByRole('button', { name: 'View data' }).click()
  await page.waitForTimeout(800)
  await page.screenshot({ path: path.join(SHOTS, '04-data-table.png') })
  const zoneLink = page.getByRole('table').locator('tbody .table-zone-link').first()
  if (await zoneLink.count()) {
    await zoneLink.click()
    await page.waitForTimeout(1800) // camera eases onto the selected zone
    await page.screenshot({ path: path.join(SHOTS, '05-zone-inspector.png') })
  }
  await page.getByRole('button', { name: 'Close table' }).click().catch(() => {})

  // Remaining routes.
  for (const [name, route] of [
    ['06-history', '/history'],
    ['07-status', '/status'],
    ['08-methodology', '/methodology'],
    ['09-settings', '/settings'],
    ['10-about', '/about'],
  ]) {
    await page.goto(BASE + route, { waitUntil: 'domcontentloaded' })
    await page.waitForTimeout(1400)
    await page.screenshot({ path: path.join(SHOTS, `${name}.png`) })
  }
  await context.close()

  // Mobile viewport.
  const mobile = await browser.newContext({ viewport: { width: 390, height: 844 } })
  const mobilePage = await mobile.newPage()
  await mobilePage.goto(BASE + '/', { waitUntil: 'domcontentloaded' })
  await closeTutorial(mobilePage)
  await mobilePage.waitForTimeout(4000)
  await mobilePage.screenshot({ path: path.join(SHOTS, '11-mobile.png') })
  await mobile.close()
  await browser.close()
}

async function walkthroughVideo() {
  const browser = await chromium.launch()
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: path.join(ROOT, 'media'), size: { width: 1280, height: 800 } },
  })
  const page = await context.newPage()
  await page.goto(BASE + '/', { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(2200)

  // The guided tutorial, end to end.
  for (let step = 0; step < 5; step++) {
    await page.getByRole('button', { name: step === 4 ? 'Enter HF' : 'Next' }).click()
    await page.waitForTimeout(1100)
  }

  // The globe, population dots, and terrain after the entry flight lands.
  await page.waitForTimeout(3000)

  // Model registry: the live weekly-retrained model, then back to the map.
  await page.getByRole('button', { name: /forecast model: LIVE/ }).click()
  await page.waitForTimeout(1400)
  await page.getByRole('button', { name: 'Close model registry' }).click()
  await page.waitForTimeout(800)

  // Compare an archived catalog model, then return to the live model.
  await page.getByRole('button', { name: /forecast model: LIVE/ }).click()
  await page.waitForTimeout(900)
  const citadel = page.getByRole('button', { name: /CITADEL/ })
  if (await citadel.count()) await citadel.click()
  await page.waitForTimeout(1600)
  await page.getByRole('button', { name: /forecast model: CITADEL/ }).click()
  await page.waitForTimeout(900)
  await page.getByRole('button', { name: /Live registry model/ }).first().click()
  await page.waitForTimeout(1200)

  // Data table, zone inspector, and a slow zoom for the closing shot.
  await page.getByRole('button', { name: 'View data' }).click()
  await page.waitForTimeout(1000)
  const zoneLink = page.getByRole('table').locator('tbody .table-zone-link').first()
  if (await zoneLink.count()) {
    await zoneLink.click()
    await page.waitForTimeout(2000)
  }
  await page.mouse.move(640, 400)
  for (let i = 0; i < 6; i++) {
    await page.mouse.wheel(0, -160)
    await page.waitForTimeout(280)
  }
  await page.waitForTimeout(1600)

  await context.close() // finalises the video file
  await browser.close()
}

await heroShots()
await walkthroughVideo()
console.log('media captured in', path.join(ROOT, 'media'))
