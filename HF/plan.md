# Humanitarian Forecaster product and technical plan

## Mission
Humanitarian Forecaster (HF) is an Ethiopia-centered analytical instrument for interpreting probabilistic conflict-location research. Its public interface must make uncertainty, evidence quality, cutoff time, and model limitations as visible as the forecast itself. It is not a tactical coordinate product, verified front line, safe-route service, or evacuation order.

## Product principles
1. The globe is the product, not a decorative hero. Ethiopia and the Horn remain the geographic center of gravity.
2. Every pixel represents human consequence. Visual emphasis communicates analytical meaning rather than spectacle.
3. Forecasts are zones and ranked possibilities, never declarations. Observations, source signals, inferences, forecasts, uncertainty, and unknown outcomes use separate marks and language.
4. Fine coordinates are coarsened before publication when they could expose vulnerable people, shelters, sources, or movement.
5. The complete map state has a keyboard-operable, screen-reader-compatible table and textual summary.
6. Production failure is explicit. Demo data is synthetic and labeled; the application never silently substitutes it for live results.

## Architecture
- **Client:** React + TypeScript + Vite, React Router, Tailwind CSS, Motion, MapLibre GL JS, Zod.
- **Map:** one MapLibre WebGL context using globe projection and native GeoJSON, heatmap, fill, line, circle, and fill-extrusion layers. No parallel R3F renderer. An SVG regional fallback preserves analysis without WebGL.
- **API:** FastAPI + Pydantic deployed as Vercel Python Functions. Static reads are cacheable; signed writes fail closed without production configuration.
- **Model serving:** ONNX Runtime CPU behind `feature-contract.v1`. PyTorch remains the offline parity oracle. The serving package records model, feature, and artifact checksums.
- **State:** PostgreSQL for transactional forecast/idempotency/adapter state; Upstash-compatible Redis REST for distributed, endpoint-cost rate limits; immutable object storage for artifacts.
- **Ingestion:** private GitHub Actions gathers six allowlisted sources, performs schema-constrained Groq extraction, constructs cutoff-safe features, and signs an idempotent request to Vercel.
- **Learning:** an idempotent third-day reconciliation updates only a norm-bounded calibration adapter from mature observed outcomes and promotes it only through replay gates.

## Experience model
Desktop uses a narrow navigation rail, dominant globe, attached evidence inspector, and bottom timeline. Mobile uses a compact context bar, map, accessible bottom sheet, timeline, and bottom navigation. Layers are selectable rather than piled on simultaneously. Core views are Forecast, History, Methodology, Policy, Status, and About.

The visual language uses mineral surfaces, deep ink, restrained teal probability, ochre uncertainty, and crimson only for observed conflict. Typography combines an editorial serif for interpretation with a highly legible sans for controls and data. Motion explains state changes and respects reduced-motion preferences.

## Delivery phases
1. Establish versioned contracts, project configuration, fixtures, demo/production provider boundary, and secure headers.
2. Build the accessible responsive shell and informational routes.
3. Build globe layers, fallback map, legends, filters, selection, and table parity.
4. Build timeline, evidence inspection, comparison, provenance, source health, and uncertainty explanation.
5. Build FastAPI reads/writes, security boundaries, storage adapters, rate classes, and learning reconciliation.
6. Build and verify the typed ONNX serving bridge against the existing PyTorch inference path.
7. Add GitHub ingestion, documentation, threat model, operating runbooks, and configuration handoff.
8. Validate, visually inspect multiple viewports and states, harden, commit, and push the feature branch.

## Key risks and controls
- **False precision:** 20 km advisory circles, rounded public coordinates, explicit horizon/cutoff, calibration and candidate-ceiling context.
- **Sensitive disclosure:** allowlisted aggregate fields, server-side coarsening, excerpt limits, log redaction, no raw-source publication.
- **Model drift:** immutable base, bounded adapter, mature labels, replay gate, atomic rollback, versioned provenance.
- **Pipeline replay/forgery:** body hash, HMAC, constant-time comparison, freshness window, durable run-ID idempotency, transaction locks.
- **Source failure/injection:** independent source health, bounded retries, content treated only as untrusted data, strict Groq schema validation.
- **WebGL/performance:** lazy secondary layers, geometry limits, adaptive quality, context-loss handling, SVG/table fallback.
- **Accessibility:** semantic routes and landmarks, visible focus, non-color marks, keyboard map alternatives, live states, reduced motion.

## Acceptance criteria
- The opening view is an Ethiopia-centered working globe with distinct forecast, uncertainty, evidence, observation, terrain, momentum, and exposure layers.
- Forecast selection, timeline, legend, provenance, and evidence work with pointer and keyboard; all visible data is available in a table.
- Demo and production modes are visibly and structurally isolated.
- Write routes reject unsigned, stale, replayed, oversized, or invalid requests and production writes reject incomplete infrastructure configuration.
- No secret enters the browser bundle or repository; logs redact security and sensitive-source fields.
- Frontend lint/type/tests/build, backend lint/type/tests/startup, schema parity, model parity, accessibility checks, palette validation, security checks, and responsive visual inspection pass.
- The deployed documentation explains setup, limitations, methodology, policy, incident response, and required secret-store configuration.

## Rejected alternatives
- **Next.js:** server components add little to a WebGL-first analytical client and couple frontend rendering to the serving runtime.
- **React Three Fiber globe:** duplicates MapLibre camera/render ownership and complicates accessibility and performance for layers MapLibre already supports.
- **Full PyTorch in Vercel:** avoidable cold-start and bundle cost; ONNX Runtime is the bounded serving target.
- **News-as-reward RL:** methodologically invalid. News is input; only later mature observations can supervise an update.
- **Animation component kit as design system:** inappropriate for a restrained humanitarian instrument; copied components are allowed only when they solve a real interaction accessibly.
