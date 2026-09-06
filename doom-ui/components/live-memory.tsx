'use client';
import {memo} from 'react';
import type {Live} from '@/lib/live';
import {PixelIcon} from '@/components/pixel-icon';
export const LiveMemory=memo(function LiveMemory({data,live}:{data:Live;live:boolean}){
 const m=data.learning;if(!m)return null;
 const max=Math.max(1,...m.efficacy_histogram);
 const rounds=(data.episodes??[]).filter(r=>typeof r.tick==='number').slice(-12);
 const longest=Math.max(1,...rounds.map(r=>r.tick!/35));
 return <section className="live-memory" aria-label="Experimental training measurements">
  <div className="panel-heading"><h2><PixelIcon name="brain"/> MEMORY / {m.enabled?'TRAINING':'FROZEN'}</h2><span className="panel-caption">{live?'LIVE':'LAST RECEIVED'} · UNVALIDATED</span></div>
  <div className="memory-metrics"><div><span>CHANGED WEIGHTS</span><strong>{m.changed_edges.toLocaleString()}<small> / {m.plastic_edges.toLocaleString()}</small></strong></div><div><span>AVERAGE STRENGTH</span><strong>{m.mean_efficacy.toFixed(3)}×</strong></div><div><span>DAMAGE FEEDBACK</span><strong>{(m.delivered_ms/1000).toFixed(2)}<small> s delivered</small></strong></div></div>
  <div className="memory-plots"><div><h3>SYNAPSE STRENGTH</h3><svg viewBox="0 0 320 76" role="img" aria-label={`Distribution of ${m.plastic_edges} modeled memory connections; 1 times is their initial strength.`} shapeRendering="crispEdges">{m.efficacy_histogram.map((n,i)=><rect key={i} x={i*16} y={70-n/max*60} width="12" height={n/max*60} fill="currentColor"/>)}<path d="M151.58 0V74" stroke="currentColor" strokeDasharray="2 3"/></svg><div className="plot-key"><span>0.1×</span><span>Dotted line: initial 1×</span><span>2×</span></div></div><div><h3>RECENT ROUND SURVIVAL</h3><div className="round-bars" role="img" aria-label={rounds.length?rounds.map(r=>`Round ${r.episode}: ${(r.tick!/35).toFixed(1)} game seconds`).join('; '):'No completed rounds yet'}>{rounds.length?rounds.map(r=><div key={r.episode} title={`Round ${r.episode}: ${(r.tick!/35).toFixed(1)} s; ${r.kills} kills`}><span style={{height:`${Math.max(2,r.tick!/35/longest*56)}px`}}/><small>{r.episode}</small></div>):<span>Waiting for a completed round</span>}</div><div className="plot-key"><span>Game seconds · latest 12</span><span>Scale: {longest.toFixed(1)} s</span></div></div></div>
  <p>{m.enabled?'Plasticity is enabled.':'Weights are frozen for this control.'} Whether training improves survival is <strong>unproven</strong>. <a href="/how-it-works">How training works ↗</a></p>
 </section>;
});
