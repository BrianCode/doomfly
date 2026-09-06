'use client';
import {useState} from 'react';
import type {ReactNode} from 'react';
import {PixelIcon} from '@/components/pixel-icon';

type Trace={condition:string;replicate:number;seed:number;survival_seconds:number;points:{seconds:number;health:number;efficacy:number}[]};
type Report={status:string;generated_at:string;verdict:string;learning_demonstrated:boolean;
 gates:{name:string;passed:boolean;detail:string}[];
 vision:{stimulus:string;receptor_spikes:number;T4_spikes:number;T5_spikes:number;KC_spikes:number;MBON_spikes:number}[];
 conditioning:{condition:string;changed_edges:number;mean_efficacy:number;before_MBON_spikes:number[];after_MBON_spikes:number[];DAN_spikes:number}[];
 comparison:{condition:string;episodes:number;training_replicates:number;mean_survival_seconds:number|null;deaths:number;censored:number;changed_edges_max:number;training_KC_spikes:number;training_DAN_spikes:number;delivered_US_tics:number}[];
 traces:Trace[];performance:{episodes:number;wall_seconds:number;brain_seconds:number;aggregate_speed:number|null};
 limits:string[];sources:{title:string;url:string}[]};
const names:Record<string,string>={plastic:'PLASTICITY ON',frozen:'FROZEN',shuffled:'SHUFFLED REWARD'};
const readable=(s:string)=>s.replaceAll('_',' ');
const fmt=(n:number)=>new Intl.NumberFormat('en-US').format(n);

function HealthPlot({trace}:{trace:Trace}){
 const max=Math.max(1,...trace.points.map(p=>p.seconds));
 const points=trace.points.map(p=>`${(36+p.seconds/max*444).toFixed(1)},${(140-Math.max(0,Math.min(100,p.health))*1.2).toFixed(1)}`).join(' ');
 // An inline SVG plot needs an accessible image role; it cannot be replaced by an img tag.
 // oxlint-disable-next-line jsx-a11y/prefer-tag-over-role
 return <figure className="lab-trace"><figcaption>{names[trace.condition]}</figcaption><svg viewBox="0 0 500 180" role="img" aria-label={`${names[trace.condition]}: recorded health over ${trace.survival_seconds.toFixed(2)} simulated seconds.`}>
  <path d="M36 20V140H480" fill="none" stroke="currentColor"/>
  <path d="M36 80H480" fill="none" stroke="currentColor" strokeDasharray="2 8" opacity=".35"/>
  <polyline points={points} fill="none" stroke="currentColor" strokeWidth="2.5"/>
  <text x="1" y="25">100</text><text x="16" y="145">0</text><text x="34" y="170">0 s</text><text x="480" y="170" textAnchor="end">{max.toFixed(1)} s</text>
 </svg><span>{trace.survival_seconds.toFixed(2)} s survival · seed {trace.seed}</span></figure>;
}

export default function LearningLab({report,latest}:{report:Report;latest?:ReactNode}){
 const [stimulus,setStimulus]=useState(report.vision[0]?.stimulus??'black');
 const selected=report.vision.find(p=>p.stimulus===stimulus);
 return <>
  <section className="lab-intro"><span className="eyebrow">FULL CONNECTOME · OPEN EXPERIMENT RECORD</span><h1>LEARNING LAB</h1><div className="lab-verdict"><PixelIcon name="brain"/><strong>{report.status==='complete'?'SURVIVAL LEARNING UNPROVEN':'VALIDATION IN PROGRESS'}</strong></div><p>Pixels in. Neural activity out. Proving learned survival takes controls. The live stream now runs experimental v6 training. These earlier tests remain failed; live weight changes do not establish learned survival.</p></section>
  {latest}
  <details className="iteration-archive" open={!latest}><summary>ARCHIVED PILOT 01 · ORIGINAL FAILED BASELINE</summary>
  <section className="lab-gates" aria-label="Scientific validation gates">{report.gates.map((g,i)=>{
   const ready=i===0?report.vision.length>0:i===1?report.conditioning.length>0:report.status==='complete';
   return <div key={g.name}><span className="lab-gate-number">0{i+1}</span><h2>{g.name}</h2><strong className={g.passed?'lab-pass':'lab-fail'}>{!ready?'PENDING':g.passed?'PASS':'FAIL'}</strong><p>{g.detail}</p></div>;
  })}</section>
  <p className="lab-caption">The original pilot’s shuffled exposure was mismatched. The scheduling fix matched exposure in two follow-up trials; those still changed zero memory connections. The original failed pilot remains shown here.</p>
  <section className="lab-section" aria-labelledby="vision-title"><div className="lab-section-heading"><h2 id="vision-title"><PixelIcon name="eye"/> DOES THE SIGNAL REACH MEMORY?</h2><label>STIMULUS<select value={stimulus} onChange={e=>setStimulus(e.target.value)}>{report.vision.map(v=><option key={v.stimulus} value={v.stimulus}>{readable(v.stimulus)}</option>)}</select></label></div>
   {selected?<><div className="lab-path"><div><span>VISUAL INPUT</span><strong>{fmt(selected.receptor_spikes)}</strong><small>receptor spikes</small></div><span aria-hidden="true">·</span><div><span>MOTION CELLS</span><strong>{fmt(selected.T4_spikes+selected.T5_spikes)}</strong><small>T4 + T5 spikes</small></div><span aria-hidden="true">·</span><div><span>MEMORY CELLS</span><strong>{fmt(selected.KC_spikes)}</strong><small>Kenyon-cell spikes</small></div></div><p className="lab-caption">Separate population readouts, not a verified serial pathway. One simulated second, identical starting states. The original pilot’s model responds to pixels, but its tested motion and memory populations produce no spikes. This historical assay predates the R8 correction.</p></>:<p>Visual tests have not completed.</p>}
  </section>
  <section className="lab-section" aria-labelledby="comparison-title"><div className="lab-section-heading"><h2 id="comparison-title"><PixelIcon name="heart"/> SURVIVAL: ON VS OFF</h2><span>REAL DOOM · HEALTH GATHERING</span></div>
   <p className="lab-caption">Damaging floor. Health pickups. Fixed neural controls. Tests use new game seeds, with reinforcement and further weight changes switched off.</p>
   <div className="lab-comparison">{report.comparison.map(c=><div key={c.condition}><h3>{names[c.condition]}</h3><strong>{c.mean_survival_seconds===null?'—':c.mean_survival_seconds.toFixed(2)}<small> s</small></strong><span>mean observed test time</span><dl><div><dt>Changed memory connections</dt><dd>{fmt(c.changed_edges_max)}</dd></div><div><dt>Independent training seeds</dt><dd>{c.training_replicates}</dd></div><div><dt>Test episodes / timeouts</dt><dd>{c.episodes} / {c.censored}</dd></div></dl></div>)}</div>
   {report.traces.length?<><div className="lab-traces">{report.traces.map(t=><HealthPlot key={t.condition} trace={t}/>)}</div><p className="lab-caption">Recorded test episodes from one matched seed pair. Overlapping outcomes are shown as measured. All held-out episodes reached the 12 s cap; their eventual survival time is unknown. This small pilot cannot establish a survival benefit.</p></>:<p className="lab-caption">The experiment is still collecting matched test episodes. No learning curve is implied.</p>}
  </section>
  <details className="disclosure"><summary><span><PixelIcon name="brain"/> THE MEMORY RULE</span><PixelIcon name="plus"/></summary><div className="disclosure-body"><p>Recent activity leaves a fading trace at a Kenyon cell’s existing output connections. Spikes from identified PPL101 dopamine cells can weaken eligible connections onto MBON11. These are 4,184 existing connections inside the full retained male connectome.</p><p>The equations and parameters are an experimental approximation. Dopamine delivery is modeled separately from fast excitation for these two cells. The mapping from game damage to stimulation is engineered; it is not a validated pain pathway.</p><div className="table-scroll"><table><thead><tr><th>Conditioning control</th><th>Changed connections</th><th>Efficacy remaining</th><th>Dopamine-cell spikes</th></tr></thead><tbody>{report.conditioning.map(c=><tr key={c.condition}><td>{readable(c.condition)}</td><td>{fmt(c.changed_edges)}</td><td>{Math.round(c.mean_efficacy*100)}%</td><td>{fmt(c.DAN_spikes)}</td></tr>)}</tbody></table></div><p>These tests directly stimulate memory cells and bypass vision. The full-network candidate loses cue specificity; changes also occur without imposed reward. Numerical rule tests passing does not make this a valid reproduction of fly conditioning.</p></div></details>
  <details className="disclosure"><summary><span><PixelIcon name="code"/> FAST DEV / EXACT CLOCKS</span><PixelIcon name="plus"/></summary><div className="disclosure-body"><p>Development runs omit browser updates, video compression and real-time pacing. They still generate every sensory frame and integrate every 0.1 ms neural timestep. Independent seed replicas can run in separate processes. Brain checkpoints preserve the complete model state. Serial and parallel runs produced identical recorded inputs, spikes and controls in the scheduling checks. The neural kernel remains the main compute cost.</p><p>{report.performance.episodes?`${report.performance.episodes} episodes processed ${report.performance.brain_seconds.toFixed(1)} seconds of brain time in ${report.performance.wall_seconds.toFixed(1)} seconds of wall time (${report.performance.aggregate_speed?.toFixed(2)}× aggregate speed).`:'Performance measurements will appear when the pilot completes.'} These measurements include competition with other work on the simulation host.</p></div></details>
  <details className="disclosure"><summary><span><PixelIcon name="signal"/> EVIDENCE + LIMITS</span><PixelIcon name="plus"/></summary><div className="disclosure-body"><ul>{report.limits.map(l=><li key={l}>{l}</li>)}</ul><nav className="source-links" aria-label="Primary scientific sources">{report.sources.map(s=><a key={s.url} href={s.url} target="_blank" rel="noreferrer">{s.title} ↗</a>)}</nav></div></details>
  <footer className="lab-footer"><a href="/learning-results.json" download="learning-results.json">Download results ↗</a><a href="/learning-source.zip" download="learning-source.zip">Experiment source ↗</a><a href="/">Watch experimental training ↗</a></footer>
  </details>
 </>;
}
