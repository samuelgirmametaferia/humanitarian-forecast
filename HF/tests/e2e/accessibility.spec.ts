import AxeBuilder from '@axe-core/playwright'
import { expect, test } from 'playwright/test'

const viewports = [
  { name: 'desktop', width: 1440, height: 900 },
  { name: 'laptop', width: 1180, height: 760 },
  { name: 'tablet', width: 820, height: 1024 },
  { name: 'mobile', width: 390, height: 844 },
]

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    if (!localStorage.getItem('hf-preferences')) {
      localStorage.setItem('hf-preferences', JSON.stringify({ state: { tutorialSeen: true }, version: 0 }))
    }
  })
})

test('forecast shell passes automated accessibility checks', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Humanitarian Forecaster' })).toBeVisible()
  await page.getByRole('button', { name: 'Accessible Map' }).click()
  const results = await new AxeBuilder({ page }).analyze()
  expect(results.violations).toEqual([])
})

test('skip link moves focus to the forecast', async ({ page }) => {
  await page.goto('/')
  await page.keyboard.press('Tab')
  await expect(page.getByRole('link', { name: 'Skip to main content' })).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page.locator('#main-content')).toBeFocused()
})

test('data table exposes every forecast zone', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'View data' }).click()
  const table = page.getByRole('table')
  await expect(table).toBeVisible()
  await expect(table.locator('tbody tr')).toHaveCount(5)
  await page.getByRole('button', { name: 'Probability' }).click()
  await page.getByRole('button', { name: 'Close table' }).click()
  await expect(table).toBeHidden()
})

test('map mode is keyboard accessible and persisted', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Accessible Map' }).click()
  const zone = page.getByRole('button', { name: /Select .*rank 2/ })
  await zone.focus()
  await page.keyboard.press('Enter')
  await expect(zone).toHaveClass(/is-selected/)
  await page.reload()
  await expect(page.getByRole('button', { name: 'Accessible Map' })).toHaveAttribute('aria-pressed', 'true')
  await page.getByRole('button', { name: '3D Globe' }).click()
  await expect(page.getByRole('button', { name: '3D Globe' })).toHaveAttribute('aria-pressed', 'true')
})

test('settings persist and legal documents open in new tabs', async ({ page, context }) => {
  await page.goto('/settings')
  await page.getByLabel('Theme').selectOption('light')
  await page.getByLabel('Reduced motion').check()
  await page.reload()
  await expect(page.getByLabel('Theme')).toHaveValue('light')
  await expect(page.getByLabel('Reduced motion')).toBeChecked()
  const popup = context.waitForEvent('page')
  await page.getByRole('link', { name: 'Privacy Policy' }).click()
  await expect((await popup).getByRole('heading', { name: 'Privacy Policy' })).toBeVisible()
})

test('tutorial can be replayed from settings', async ({ page }) => {
  await page.goto('/settings')
  await page.getByRole('button', { name: 'Replay tutorial' }).click()
  await expect(page).toHaveURL('/')
  await expect(page.getByRole('dialog', { name: 'See pressure building before it becomes a surprise.' })).toBeVisible()
  await page.getByRole('button', { name: 'Next' }).click()
  await expect(page.getByRole('heading', { name: 'Start with the places, not the model.' })).toBeVisible()
  await page.getByRole('button', { name: 'Close tutorial' }).click()
})

test('model registry loads a different published prediction', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Accessible Map' }).click()
  await page.getByRole('button', { name: /forecast model: NIGHTJAR/ }).click()
  const registry = page.getByRole('dialog', { name: 'Select forecast model' })
  await expect(registry).toBeVisible()
  await expect(registry.locator('.model-list > button')).toHaveCount(33)
  await registry.getByRole('button', { name: /CITADEL/ }).click()
  await expect(page.getByRole('button', { name: /forecast model: CITADEL/ })).toBeVisible()
  await page.getByRole('button', { name: 'View data' }).click()
  await expect(page.getByRole('table').locator('tbody tr')).toHaveCount(1)
})

test('model selector and map recovery control remain available on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await expect(page.getByRole('button', { name: /forecast model: NIGHTJAR/ })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Focus map on Ethiopia' })).toBeVisible()
})

for (const viewport of viewports) {
  test(`${viewport.name} viewport has no horizontal page overflow`, async ({ page }) => {
    await page.setViewportSize(viewport)
    await page.goto('/')
    await page.getByRole('button', { name: 'Accessible Map' }).click()
    const overflow = await page.locator('html').evaluate((element: HTMLElement) => element.scrollWidth - element.clientWidth)
    expect(overflow).toBeLessThanOrEqual(1)
    await expect(page.getByRole('navigation', { name: 'Primary navigation' })).toBeVisible()
    await expect(page.getByRole('region', { name: 'Accessible regional forecast map' })).toBeVisible()
    await page.screenshot({ path: `test-results/browser/${viewport.name}.png`, fullPage: true })
  })
}
