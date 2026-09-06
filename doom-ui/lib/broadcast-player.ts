import {parseLive,type Live} from './live';
export const PLAYOUT_DELAY_MS=2500;
type Index={transport:1;status:'running';run_id:string;generated_at_ms:number;segments:string[];playout_delay_ms:number;phase:'baseline'|'training'|'frozen-control'};
export function parseBroadcastIndex(value:unknown):Index{
 const d=value as Index;
 if(!d||d.transport!==1||d.status!=='running'||!Number.isFinite(d.generated_at_ms)||!/^[-a-f0-9]{36}$/.test(d.run_id)||!Array.isArray(d.segments)||d.segments.length>4||!d.segments.every(s=>/^[0-9]{10,14}$/.test(s))||d.playout_delay_ms!==PLAYOUT_DELAY_MS||!['baseline','training','frozen-control'].includes(d.phase))throw new Error('Invalid broadcast index');
 return d;
}
export function parseSegment(value:unknown,run:string):Live[]{
 const d=value as {transport:number;shared:Record<string,unknown>&{run_id:string;retina:object;raster_ids:string[]};frames:Record<string,unknown>[]};
 if(!d||d.transport!==1||!d.shared||d.shared.run_id!==run||!Array.isArray(d.frames)||d.frames.length>35)throw new Error('Invalid segment');
 const result=d.frames.map(f=>parseLive({...d.shared,...f,retina:{...d.shared.retina,luminance:f.luminance},
  raster:{neuron_ids:d.shared.raster_ids,bins:[f.raster_bin]},populations:[]}));
 if(result.some((f,i)=>i>0&&(f.sequence<=result[i-1].sequence||f.generated_at_ms<result[i-1].generated_at_ms)))throw new Error('Out-of-order frames');
 return result;
}

export class BroadcastPlayer{
 run='';sequence=0;private queue:Live[]=[];private bins:Live['raster']['bins']=[];
 discardPending(){this.queue=[];}
 ingest(frames:Live[]){
  if(!frames.length)return;
  if(this.run!==frames[0].run_id){this.run=frames[0].run_id;this.sequence=0;this.queue=[];this.bins=[];}
  const known=new Set(this.queue.map(f=>f.sequence));
  for(const f of frames)if(f.run_id===this.run&&f.sequence>this.sequence&&!known.has(f.sequence)){this.queue.push(f);known.add(f.sequence);}
  this.queue.sort((a,b)=>a.sequence-b.sequence);this.queue=this.queue.slice(-140);
 }
 take(now:number):Live|null{
  let selected:Live|null=null;
  while(this.queue.length&&(this.queue[0].generated_at_ms<=now-PLAYOUT_DELAY_MS||(!this.sequence&&!selected))){
   selected=this.queue.shift()!;this.sequence=selected.sequence;
   for(const bin of selected.raster.bins)if(!this.bins.length||bin.neural_ms>this.bins[this.bins.length-1].neural_ms)this.bins.push(bin);
  }
  if(!selected)return null;
  this.bins=this.bins.slice(-160);
  return {...selected,raster:{...selected.raster,bins:this.bins.slice()}};
 }
}
