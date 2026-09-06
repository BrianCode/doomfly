import results from '@/public/audit-experiments.json';
export default function Review(){
 return <main className="arcade review-copy">
  <header className="masthead"><a href="/" className="brand"><img src="/fly-logo.svg?v=side-fly" width="32" height="32" alt=""/><span>DOOMFLY</span></a><a className="header-guide" href="/how-it-works">How it works</a></header>
  <article>
   <h1>REAL WIRING.<br/>APPROXIMATE BRAIN.</h1>
   <p>The released male fly connectome drives a real Doom-engine game. This is an experimental neural interface. Fly vision, natural behavior and learning have not been validated.</p>
   <div className="table-scroll"><table><thead><tr><th>Part</th><th>Finding</th></tr></thead><tbody>
    <tr><td>Connectome</td><td>All 166,700 retained cells and 25,582,938 connections checked against the source files.</td></tr>
    <tr><td>Neural solver</td><td>A refractory-period error was fixed. Both backends now pass independent Brian2 comparisons.</td></tr>
    <tr><td>Vision</td><td>Real frame samples; assumed eye geometry and spiking proxies for graded visual cells.</td></tr>
    <tr><td>Controls</td><td>Actual modeled spikes drive a fixed, engineered mapping. Background current also drives activity.</td></tr>
    <tr><td>Learning</td><td>Absent. No synaptic weights change. Optional reward stimulation is not learning.</td></tr>
   </tbody></table></div>
   <h2>DOES VISION HELP?</h2>
   <p>Three matched game seeds, six simulated seconds each. The decoder was held fixed. These short checks do not establish learned skill or a reliable performance estimate.</p>
   <div className="table-scroll"><table><thead><tr><th>Seed</th><th>Live vision kills</th><th>Black input kills</th><th>Controls off kills</th></tr></thead><tbody>
    {[41027,41028,41029].map(seed=><tr key={seed}><td>{seed}</td>{['intact','blank_vision','controls_clamped'].map(condition=><td key={condition}>{results.closed_loop_trials.find(t=>t.seed===seed&&t.condition===condition)?.kills??'—'}</td>)}</tr>)}
   </tbody></table></div>
   <p>Black input still allows movement and firing. Removing the artificial lamina background current silences the selected controls in the fixed-frame test. Image sensitivity alone does not demonstrate recognition or aiming.</p>
   <h2>WHAT WAS CORRECTED?</h2>
   <p>Refractory signal handling, unequal time bins in the activity plot, misleading eye labels, stimulation timing in the audit log, and source/binary checks. Game assets are now explicitly selected and fingerprinted. Previous validation files are marked as legacy.</p>
   <h2>WHAT COMES NEXT?</h2>
   <p>Calibrate graded visual dynamics against real flash and motion responses. Validate a specific sensory-to-action pathway. Then implement and test a biologically grounded memory mechanism, with controlled reward and held-out evaluation.</p>
   <nav className="source-links" aria-label="Audit evidence"><a href="/neuroscience-review.md">Full review ↗</a><a href="/data-integrity.json">Every-edge audit ↗</a><a href="/official-source-check.json">Official source checks ↗</a><a href="/audit-experiments.json">Experiment results ↗</a><a href="https://github.com/nftechie/doomfly" target="_blank" rel="noopener noreferrer">Source + tests ↗</a><a href="/">Back to live experiment →</a></nav>
   <p className="detail-intro">Reviewed September 5, 2026. One male reconstruction. Live streaming depends on the simulation host remaining online.</p>
  </article>
 </main>;
}
