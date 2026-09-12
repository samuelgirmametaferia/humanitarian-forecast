export function PrivacyRoute() {
  return <main id="main-content" className="document-route"><article><p className="document-kicker">HF</p><h1>Privacy Policy</h1><p className="document-date">Effective 12 September 2026</p>
    <h2>Overview</h2><p>HF is a public research interface. You can view forecasts without creating an account or submitting personal information.</p>
    <h2>Local preferences</h2><p>Theme, map, motion, layer, and density choices are stored in your browser. HF does not receive these preferences through the application API. You can remove them by clearing site data.</p>
    <h2>Forecast requests</h2><p>The browser requests forecast and service-status data from the HF API. The application does not include forms for names, email addresses, credentials, or payment details.</p>
    <h2>Infrastructure records</h2><p>Hosting providers may process standard request information, such as IP address, browser details, requested URL, and request time, to deliver and protect the service.</p>
    {/* TODO(legal): Confirm the deployed hosting provider, log retention, analytics, and contact channel before public release. */}
    <h2>Third-party map data</h2><p>The interactive globe can request terrain tiles from the map provider identified in the map attribution. The accessible map does not require those terrain tiles.</p>
    <h2>Changes</h2><p>This policy will be updated when the application’s data practices change.</p>
  </article></main>
}
