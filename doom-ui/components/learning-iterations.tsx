import {PixelIcon} from '@/components/pixel-icon';

type Episode={seed:number;condition:string;phase:string;survival_seconds:number;right_censored:boolean};
type Iterations={generated_at:string;candidates:number;
 conditioning:{rows:{plus:number;condition:string;suppression:number[]}[]};
 stability:{rows:{cue:string;baseline:{DAN_hz:number[]};recovery:{DAN_hz:number[]}}[]};
 survival:{groups:{condition:string;mean_observed_survival:number;deaths:number;censored:number;changed_edges:number;test_episodes:number;training_replicates:number}[];episodes:Episode[];horizon_seconds:number};
 performance:{brain_seconds:number;wall_seconds:number;kernel_seconds:number};
 limits:string[];sources:{title:string;url:string}[]};
const names:Record<string,string>={plastic:'LEARNING ON',frozen:'FROZEN',shuffled:'SHUFFLED TIMING'};
const number=(n:number)=>new Intl.NumberFormat('en-US').format(n);

export default function LearningIterations({report}:{report:Iterations}){
 const left=report.conditioning.rows.filter(r=>r.plus===0&&['paired','no_imposed_US'].includes(r.condition));
 const checks=report.survival.episodes.filter(r=>r.condition==='plastic'&&['held_out','vision_black','memory_erased','retention_5s'].includes(r.phase));
 const phase:Record<string,string>={held_out:'Unseen start',vision_black:'Vision blacked out',memory_erased:'Memory erased',retention_5s:'After 5 s rest'};
 return <div className="iteration-record">
  <div className="iteration-strip"><span>{report.candidates} CANDIDATES TESTED</span><span>166,700 NEURONS RETAINED</span><span>{report.generated_at.slice(0,10)}</span></div>
  <section className="lab-section" aria-labelledby="control-title">
   <div className="lab-section-heading"><h2 id="control-title"><PixelIcon name="brain"/> THE CONTROL CAUGHT IT</h2><span>CANDIDATE 05 · FAIL</span></div>
   <p className="iteration-lead">The cue response changed. Punishment didn’t explain it.</p>
   <div className="control-bars">{left.map(r=><div key={r.condition}><span>{r.condition==='paired'?'CUE + PUNISHMENT':'CUE ALONE'}</span><div className="control-track" aria-hidden="true"><i style={{width:`${Math.max(0,Math.min(100,r.suppression[0]*100))}%`}}/></div><strong>{(r.suppression[0]*100).toFixed(1)}%</strong></div>)}</div>
   <p className="lab-caption">Reduction in the trained cue’s MBON response, after training the left cue. The same reduction without imposed punishment fails the reinforcement control. The counterbalanced right-cue experiment also failed. Changed connections alone do not establish learning.</p>
  </section>
  <section className="lab-section" aria-labelledby="new-survival-title">
   <div className="lab-section-heading"><h2 id="new-survival-title"><PixelIcon name="heart"/> DOES MEMORY BUY MORE TIME?</h2><span>CANDIDATE 06 · REAL DOOM</span></div>
   <p className="lab-caption">Blue floor causes engine damage. Gray floor is safe. Live pixels drive the full neural graph and the same fixed button mapping in every arm. Test rounds freeze memory and turn off punishment stimulation.</p>
   <div className="lab-comparison">{report.survival.groups.map(g=><div key={g.condition}><h3>{names[g.condition]}</h3><strong>{g.mean_observed_survival.toFixed(2)}<small> s</small></strong><span>mean observed survival</span><dl><div><dt>Changed connections</dt><dd>{number(g.changed_edges)}</dd></div><div><dt>Training replicas / test starts</dt><dd>{g.training_replicates} / {g.test_episodes}</dd></div><div><dt>Deaths / capped rounds</dt><dd>{g.deaths} / {g.censored}</dd></div></dl></div>)}</div>
   <p className="lab-caption">Exploratory pilot, capped at {report.survival.horizon_seconds} s. One training replica is insufficient to establish a repeatable benefit. Capped rounds have unknown eventual survival. Physiological and conditioning checks still fail.</p>
   <details className="disclosure"><summary><span>TEST STARTS + MEMORY ERASURE</span><PixelIcon name="plus"/></summary><div className="disclosure-body"><div className="table-scroll"><table><thead><tr><th>Check</th><th>Start seed</th><th>Observed survival</th><th>End</th></tr></thead><tbody>{checks.map(r=><tr key={`${r.phase}-${r.seed}`}><td>{phase[r.phase]}</td><td>{r.seed}</td><td>{r.survival_seconds.toFixed(2)} s</td><td>{r.right_censored?'Time cap':'Death'}</td></tr>)}</tbody></table></div><p>Blacking out vision and erasing learned connections test what behavior depends on. Five seconds of simulated rest is a short technical check, not evidence of lasting biological memory.</p></div></details>
  </section>
  <section className="lab-section" aria-labelledby="remaining-title"><div className="lab-section-heading"><h2 id="remaining-title"><PixelIcon name="eye"/> VISION REACHES MEMORY. RECOVERY FAILS.</h2></div>
   <p className="lab-caption">Adding the documented R8 → aMe12 visual pathway made a blue image activate 14 visual memory cells in an early one-second assay; black activated none. The later model still stays too active after the image disappears.</p>
   <div className="table-scroll"><table><thead><tr><th>Image shown</th><th>Dopamine cells before</th><th>3 s after removal</th></tr></thead><tbody>{report.stability.rows.filter(r=>r.cue!=='black').map(r=><tr key={r.cue}><td>{r.cue.replaceAll('_',' ')}</td><td>{r.baseline.DAN_hz.join(' / ')} Hz</td><td>{r.recovery.DAN_hz.join(' / ')} Hz</td></tr>)}</tbody></table></div>
   <p className="lab-caption">Two identified dopamine cells, measured separately. These are simulated spike rates. Published resting activity helped calibrate their starting rates; persistent cue-triggered activity remains unvalidated. Restoring a pathway does not validate the whole visual system.</p>
  </section>
  <details className="disclosure"><summary><span><PixelIcon name="code"/> MODEL CHANGES + EXACT CLOCKS</span><PixelIcon name="plus"/></summary><div className="disclosure-body"><p>All six candidates retain the same male reconstruction. New models separate annotated modulators from generic fast excitation, add a documented R8 cotransmission route, and test KC adaptation and a dopamine-centered memory rule on 4,184 existing KC → MBON11 connections. The rule is adapted from published work; transfer to this full network is unvalidated.</p><p>The runner processes every 35 Hz Doom frame and every 0.1 ms neural timestep. A health decrease schedules exactly 200 ms of stimulation into identified dopamine cells, beginning next game tic. That feedback is engineered. Health and coordinates never select an action.</p><p>Recorded episodes processed {report.performance.brain_seconds.toFixed(1)} brain seconds in {report.performance.wall_seconds.toFixed(1)} wall seconds ({(report.performance.brain_seconds/report.performance.wall_seconds).toFixed(2)}×). Neural integration accounted for {(100*report.performance.kernel_seconds/report.performance.wall_seconds).toFixed(0)}% of that time on the shared host. Faster-than-real-time full-brain training has not been achieved.</p></div></details>
  <details className="disclosure"><summary><span><PixelIcon name="signal"/> EVIDENCE + LIMITS</span><PixelIcon name="plus"/></summary><div className="disclosure-body"><ul>{report.limits.map(l=><li key={l}>{l}</li>)}</ul><nav className="source-links" aria-label="Sources for current candidates">{report.sources.map(s=><a key={s.url} href={s.url} target="_blank" rel="noreferrer">{s.title} ↗</a>)}</nav></div></details>
  <footer className="lab-footer"><a href="/learning-iterations.json" download="learning-iterations.json">Current results ↗</a><a href="/learning-iterations-source.zip" download="learning-iterations-source.zip">Sources + raw records ↗</a><a href="/learning-iterations-source-manifest.json">Verify file hashes ↗</a></footer>
 </div>;
}
