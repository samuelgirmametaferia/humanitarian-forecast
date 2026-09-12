# Media

Screenshots and a UI walkthrough of the deployed Humanitarian Forecaster,
captured from the production site with `node scripts/capture_media.mjs`.

- `hf-walkthrough.mp4` — the five-step tutorial, the sweep-in-from-space
  globe entry, the live model registry (weekly retrained champion, registry
  history, catalog), a model comparison, the data table with its zone
  inspector, and a closing zoom.
- `hf-walkthrough.webm` — the same recording before H.264 conversion.
- `screenshots/01-tutorial.png` — first-visit tutorial over the globe.
- `screenshots/02-globe-forecast.png` — the 3D globe with forecast zones,
  WorldPop population density dots, and terrain.
- `screenshots/03-model-registry.png` — the model registry picker.
- `screenshots/04-data-table.png` — the keyboard-accessible forecast table.
- `screenshots/05-zone-inspector.png` — a selected zone with its evidence.
- `screenshots/06-history.png` … `10-about.png` — the History, Status,
  Methodology, Settings, and About pages.
- `screenshots/11-mobile.png` — the mobile layout.

Screenshots are 1440×900 (mobile: 390×844). Regenerate everything with
`node scripts/capture_media.mjs [base-url]` from the `HF` directory.
