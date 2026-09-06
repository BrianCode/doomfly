import type {Metadata} from 'next';
import {PixelIcon} from '@/components/pixel-icon';
import {SiteHeader} from '@/components/site-header';
import {VisionDiagram,NeuronDiagram,ControlsDiagram,MemoryDiagram} from '@/components/pipeline-diagrams';

export const metadata:Metadata={
 title:'How it works · DOOMFLY',
 description:'Real fly wiring, simulated neurons, one cursed controller. A short guide to the science behind DOOMFLY.',
 alternates:{canonical:'https://doomfly.example/how-it-works'},
 openGraph:{title:'How it works · DOOMFLY',url:'https://doomfly.example/how-it-works',images:['https://doomfly.example/og.png']},
};

export default function HowItWorks(){
 return <main className="arcade how-page">
  <SiteHeader page="how-it-works"/>
  <section className="how-intro" aria-labelledby="how-title">
   <div><span className="eyebrow">THE 60-SECOND EXPLANATION</span><h1 id="how-title">FLY BRAIN.<br/>DOOM PRIVILEGES.</h1><p>A real fly’s wiring diagram. Simulated neurons.<br className="desktop-break"/> One extremely cursed controller.</p></div>
   <div className="specimen-stamp"><img src="/fly-logo.svg?v=side-fly" width="96" height="96" alt=""/><span>166,700 NEURONS</span><small>ONE MALE RECONSTRUCTION</small></div>
  </section>
  <ol className="pipeline-steps" aria-label="From game frame to game action">
   <li><div className="step-heading"><span>01</span><h2>PIXELS GO IN</h2><PixelIcon name="eye"/></div><figure><VisionDiagram/></figure><p>Each Doom frame drives <strong>3,335 brightness inputs plus 811 R8 color inputs</strong>. Sample locations and RGB responses are inferred proxies. The eye plot shows the brightness samples.</p><span className="step-tag">LIVE FRAME → SENSORY INPUT</span></li>
   <li><div className="step-heading"><span>02</span><h2>NEURONS FIRE</h2><PixelIcon name="brain"/></div><figure><NeuronDiagram/></figure><p>The MaleCNS v1.0 wiring links simulated cells through <strong>25.6 million retained connections</strong>. Signals push voltages up or down; crossing a threshold fires a spike.</p><span className="step-tag">REAL WIRING / APPROXIMATE DYNAMICS</span></li>
   <li><div className="step-heading"><span>03</span><h2>DOOM GETS INPUT</h2><PixelIcon name="target"/></div><figure><ControlsDiagram/></figure><p>A fixed mapping turns selected neurons’ activity into <strong>turn, move and fire</strong>. The game advances. Its next frame starts the loop again.</p><span className="step-tag">NEURON ACTIVITY → BUTTONS</span></li>
  </ol>
  <div className="loop-caption"><span aria-hidden="true">↳</span><span>NEXT FRAME. SAME BRAIN. REPEAT.</span><small>Diagrams are schematics, not live data.</small></div>
  <section className="how-arena" aria-labelledby="arena-title"><h2 id="arena-title">MONSTERS DON’T CLOCK OUT.</h2><p>Four monsters start the round. More arrive, up to four imps and four bullet-firing zombies. Separate caps keep slow survivors from blocking the other type’s replacements. Ammo boxes respawn after collection; kills also drop ammo. Death resets the arena and ammunition. No time limit.</p><p>The neural state carries on between rounds. These are engineered Doom rules; the fly still supplies every turn, move and shot through the same fixed neural mapping. Arena v1 allowed a circling stalemate; v2’s results are recorded separately. Longer survival alone is not evidence of learning.</p></section>
  <section className="learning-note" aria-labelledby="learning-title"><div className="learning-badge">DMG<br/><span>→</span><br/>MEMORY</div><div><h2 id="learning-title">TAKE A HIT. UPDATE THE WIRING.</h2><MemoryDiagram/><p>Nonfatal damage schedules a <strong>200 ms artificial stimulus into two PPL101 dopamine cells</strong>. Their spikes and recent Kenyon-cell activity update the strength of <strong>4,184 existing KC→MBON11 connections</strong>. Game health never chooses actions or directly sets a weight.</p><p>The rule can strengthen or weaken connections; its timing traces, bounds and gradual decay are modeling choices informed by <a href="https://doi.org/10.1038/s41586-024-07819-w" target="_blank" rel="noreferrer">fly memory research</a>. Other connection strengths and the button mapping stay fixed. Memory and neural state carry across rounds. Terminal damage is excluded, and pending feedback stops at reset.</p><p><strong>Experimental training is running. Getting better is unproven.</strong> The earlier candidate failed conditioning and visual-recovery checks, and its small survival pilot included worse outcomes after training. Changed weights and longer rounds alone are not evidence of learning. <a href="/learning">See the failed tests and controls ↗</a> · <a href="/live-training.json">Exact live protocol ↗</a></p></div></section>
  <div className="how-footnote"><p>Real fly visual cells often use graded signals; ours use simplified spikes. This is an inspectable neural experiment. Realistic fly behavior remains unvalidated.</p><a href="/review">Read the scientific audit ↗</a></div>
  <div className="how-exit"><a href="/" className="watch-button"><PixelIcon name="screen"/> Back to the experiment <span aria-hidden="true">→</span></a><span>Same shared run. About 2.5 seconds of broadcast delay.<br/>Up to 8 actual captures per second; no invented frames.</span></div>
 </main>;
}
