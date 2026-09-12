import { ContentRoute } from './ContentRoute'

export function AboutRoute() {
  return <ContentRoute title="About HF">
    <section><h2>What is HF?</h2><p>HF is an independent research project that studies whether historical conflict records and public reporting can help rank where a recorded event may occur next.</p></section>
    <section><h2>What does it produce?</h2><p>The model produces a ranked set of broad candidate areas. The ranking is comparative: it shows which candidates receive more model support within one run. It does not confirm an event or identify a safe area.</p></section>
    <section><h2>Why build it?</h2><p>Location forecasting is difficult, uncertain, and easy to overstate. HF exists to test the method openly, measure where it fails, and present its output with enough context to inspect rather than simply trust it.</p></section>
  </ContentRoute>
}
