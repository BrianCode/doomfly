import {PixelIcon} from '@/components/pixel-icon';

// Static explanatory schematics. These are never represented as live telemetry.
const brightness=[0,0,1,2,2,1,0,1,2,3,3,2,1,2,3,3,2,1,0,1,2,2,1,0];
const shades=['#000','#555','#aaa','#fff'];
export function MemoryDiagram(){
 return <svg viewBox="0 0 640 120" className="pipeline-diagram" role="img" aria-label="Schematic: damage stimulates PPL101 dopamine neurons; their spikes and recent Kenyon-cell activity change existing Kenyon-cell to MBON11 synapse strengths." shapeRendering="crispEdges">
  <g fill="none" stroke="currentColor" strokeWidth="2"><rect x="10" y="12" width="128" height="44"/><rect x="202" y="12" width="146" height="44"/><rect x="422" y="72" width="112" height="38"/><rect x="202" y="72" width="112" height="38"/><path d="M140 34h56m-8-7 8 7-8 7M350 34h18v44m-7-7 7 7 7-7M314 91h108"/></g>
  <g fill="currentColor" textAnchor="middle"><text x="74" y="39">DAMAGE</text><text x="275" y="39">PPL101</text><text x="258" y="96">KC</text><text x="478" y="96">MBON11</text><text x="490" y="37">200 ms input</text></g><path d="M345 91h46" stroke="currentColor" strokeWidth="7"/>
 </svg>;
}
export function VisionDiagram(){
 return <svg viewBox="0 0 320 150" role="img" aria-label="Schematic: frame brightness is sampled at inferred R1–R6 photoreceptor positions." className="pipeline-diagram" shapeRendering="crispEdges">
  <rect x="8" y="22" width="116" height="88" fill="none" stroke="currentColor" strokeWidth="2"/>
  {brightness.map((level,i)=><rect key={i} x={16+i%6*17} y={30+Math.floor(i/6)*18} width="14" height="15" fill={shades[level]}/>)}
  <path d="M140 66h35m-8-8 8 8-8 8" fill="none" stroke="currentColor" strokeWidth="2"/>
  {[0,1,2,3,4,5,6,7,8].map(i=><g key={i}><rect x={197+i%3*37} y={27+Math.floor(i/3)*28} width="18" height="18" fill={shades[[1,3,2,2,3,1,0,2,1][i]]} stroke="currentColor" strokeWidth="2"/></g>)}
  <text x="66" y="139" textAnchor="middle">FRAME</text><text x="243" y="139" textAnchor="middle">LIGHT INPUT</text>
 </svg>;
}

export function NeuronDiagram(){
 return <svg viewBox="0 0 320 150" role="img" aria-label="Schematic: excitatory and inhibitory inputs combine in a model neuron. Crossing a voltage threshold produces a spike." className="pipeline-diagram" shapeRendering="crispEdges">
  <rect x="8" y="24" width="32" height="32" fill="currentColor"/><path d="M16 40h16m-8-8v16" stroke="#000" strokeWidth="2"/>
  <rect x="8" y="88" width="32" height="32" fill="none" stroke="currentColor" strokeWidth="2"/><path d="M16 104h16" stroke="currentColor" strokeWidth="2"/>
  <path d="M40 40h42v32h30M40 104h42V72m21-8 9 8-9 8" fill="none" stroke="currentColor" strokeWidth="2"/>
  <rect x="113" y="44" width="56" height="56" fill="none" stroke="currentColor" strokeWidth="2"/><rect x="129" y="60" width="24" height="24" fill="currentColor"/>
  <path d="M169 72h23m-8-8 8 8-8 8M202 106h106" fill="none" stroke="currentColor" strokeWidth="2"/>
  <path d="M202 56h106" stroke="currentColor" strokeDasharray="4 6"/>
  <path d="M202 102h20V90h15V76h15V38h8v64h48" stroke="currentColor" strokeWidth="3" fill="none"/>
  <text x="43" y="144" textAnchor="middle">INPUTS</text><text x="141" y="144" textAnchor="middle">CELL</text><text x="254" y="144" textAnchor="middle">SPIKE</text>
 </svg>;
}

export function ControlsDiagram(){
 return <div className="control-diagram" role="img" aria-label="Fixed decoder: DNp20 right-minus-left firing rate turns; summed DNpe017 firing rate moves forward; DNpe017 spikes press attack.">
  <div><span><b>DNp20</b><small>right − left</small></span><span className="mapping-arrow" aria-hidden="true">→</span><span className="diagram-key"><PixelIcon name="turn"/>TURN</span></div>
  <div><span><b>DNpe017</b><small>firing rate</small></span><span className="mapping-arrow" aria-hidden="true">→</span><span className="diagram-key"><PixelIcon name="move"/>MOVE</span></div>
  <div><span><b>DNpe017</b><small>spike</small></span><span className="mapping-arrow" aria-hidden="true">→</span><span className="diagram-key"><PixelIcon name="target"/>FIRE</span></div>
 </div>;
}
