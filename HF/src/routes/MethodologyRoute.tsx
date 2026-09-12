import { ContentRoute } from './ContentRoute'

export function MethodologyRoute() {
  return <ContentRoute title="Methodology">
    <section><h2>Inputs</h2><p>HF uses chronologically ordered conflict-event records, public humanitarian reporting, and geographic context. Each forecast is built only from information dated before that forecast.</p></section>
    <section><h2>Population context</h2><p>The population density dots come from the WorldPop 2020 UN-adjusted estimate for Ethiopia (30 arc-second grid, CC BY 4.0). Each dot stands for roughly ten thousand residents in a four-kilometre cell. Dots convey settlement density only; they are not household locations and are not used as model inputs or outcome labels.</p></section>
    <section><h2>Representation</h2><p>Recent events form a sequence. Candidate locations carry features such as past activity, recency, nearby activity, elevation, and terrain. Reporting volume can add context when the input pipeline provides it.</p></section>
    <section><h2>Forecast</h2><p>The model compares candidates against the event sequence and assigns a relative score to each one. An ensemble combines candidate distributions. A spatial selection step then avoids returning several near-identical areas. The interface shows the selected candidates as broad map zones.</p></section>
    <section><h2>Evaluation</h2><p>Recorded later events are compared with earlier forecasts after the outcome period is mature. Evaluation includes distance from the top-ranked candidate and whether any displayed candidate falls within a stated distance. Candidate coverage is measured separately from ranking performance.</p></section>
    <section><h2>Updates</h2><p>New observations can enter a later training or calibration run. A new model version should be compared with the current version before it is promoted. HF reserves a three-day evaluation window for this process, but an update appears in History only when a run actually occurs.</p></section>
    <section><h2>Uncertainty</h2><p>Uncertainty describes limited support in the model or its inputs. It is not a boundary around where an event will occur. Candidate percentages are relative allocations within a finite set, not independently calibrated probabilities of violence at each place.</p></section>
  </ContentRoute>
}
