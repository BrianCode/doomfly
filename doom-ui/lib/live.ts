export type Readout={index:number;id:string;type:string;side:string;spikes:number;rate_hz:number};
export type Live={schema:1;status:string;run_id:string;sequence:number;generated_at_ms:number;condition:string;decoder:string;frame:string;
 manifest:{neurons:number;edges:number;synaptic_contacts:number;retina_mapped:number;retina_total:number;retina_unmapped:number;uncertain_sign_neurons:number};
 clocks:{wall_seconds:number;neural_seconds:number;game_seconds:number;speed:number;brain_step_ms:number};
 game:{episode:number;tick:number;health:number;kills:number;ammo:number;score:number;finished:boolean;enemies?:number;enemies_spawned?:number;ammo_pickups?:number;ammo_spawned?:number};
 episodes:{episode:number;tick?:number;health:number;kills:number;score:number}[];
 action:{turn:number;forward:number;attack:boolean};readouts:Readout[];total_spikes:number;window_spikes:number;window_ms:number;total_action_ticks:number;
 populations:{name:string;spikes:number}[];
 retina:{uv:number[][];luminance:number[];full_sample_count:number;display_stride:number};
 raster:{neuron_ids:string[];bins:{neural_ms:number;window_ms:number;counts:number[];population_spikes:number}[]};
 audit:{tick:number;input_sha256:string;source_frame_sha256:string;spike_counts_sha256:string;requested:unknown;applied:unknown};
 reward:{mode:string;sugar_pulses:number;active:boolean;plasticity:boolean};
 learning?:{enabled:boolean;validated:false;model:string;plastic_edges:number;changed_edges:number;mean_efficacy:number;minimum_efficacy:number;maximum_efficacy:number;mean_absolute_change:number;bound_edges:number;efficacy_histogram:number[];damage_events:number;delivered_ms:number;stimulus_active:boolean;sha256:string};
 protocol:{dt_ms:number;lamina_bias_mv:number;retinal_gain_mv:number;seed:number;hosting:string;scenario?:string;episode_end?:string;phase?:string;recovery?:string|null};
};
export const fresh=(data:Live|null,now:number)=>!!data && now-data.generated_at_ms<5000 && now>=data.generated_at_ms-5000;
export function parseLive(input:unknown):Live{
 if(!input||typeof input!=='object')throw new Error('Invalid broadcast');
 const d=input as Live;
 const finite=(v:unknown)=>typeof v==='number'&&Number.isFinite(v);
 if(d.status!=='running'||d.schema!==1||!finite(d.sequence)||!finite(d.generated_at_ms)||typeof d.run_id!=='string'||typeof d.frame!=='string'||!d.frame.startsWith('data:image/jpeg;base64,')||d.frame.length>1000000)throw new Error('Invalid broadcast');
 for(const group of [d.clocks,d.game])if(!group||typeof group!=='object'||!Object.values(group).every(v=>typeof v==='boolean'||finite(v)))throw new Error('Invalid telemetry');
 if(!d.manifest||!['neurons','edges','synaptic_contacts','retina_mapped','retina_total','retina_unmapped','uncertain_sign_neurons'].every(k=>finite((d.manifest as unknown as Record<string,unknown>)[k])))throw new Error('Invalid graph metadata');
 if(!d.action||!finite(d.action.turn)||!finite(d.action.forward)||typeof d.action.attack!=='boolean'||!Array.isArray(d.readouts)||!d.readouts.every(r=>typeof r.id==='string'&&typeof r.type==='string'&&finite(r.spikes)&&finite(r.rate_hz)))throw new Error('Invalid neural outputs');
 const unit=(v:unknown)=>finite(v)&&(v as number)>=0&&(v as number)<=1;
 const count=(v:unknown)=>finite(v)&&Number.isInteger(v)&&(v as number)>=0;
 if(!d.retina||!Array.isArray(d.retina.uv)||!Array.isArray(d.retina.luminance)||d.retina.uv.length>10000||d.retina.uv.length!==d.retina.luminance.length||!d.retina.uv.every(p=>Array.isArray(p)&&p.length===2&&p.every(unit))||!d.retina.luminance.every(unit)||!count(d.retina.full_sample_count))throw new Error('Invalid retinal data');
 if(!d.raster||!Array.isArray(d.raster.neuron_ids)||d.raster.neuron_ids.length>256||!d.raster.neuron_ids.every(x=>typeof x==='string')||!Array.isArray(d.raster.bins)||d.raster.bins.length>160||!d.raster.bins.every((b,i)=>finite(b.neural_ms)&&finite(b.window_ms)&&b.window_ms>0&&b.neural_ms>=b.window_ms&&(!i||b.neural_ms>d.raster.bins[i-1].neural_ms)&&Array.isArray(b.counts)&&b.counts.length===d.raster.neuron_ids.length&&b.counts.every(count))||!d.audit||!d.reward||!d.protocol||!count(d.total_spikes)||!count(d.total_action_ticks)||!finite(d.window_ms)||d.window_ms<=0)throw new Error('Invalid experiment record');
 if(d.protocol.phase==='training'&&(!d.learning?.enabled||d.reward.plasticity!==true))throw new Error('Missing training evidence');
 if(d.learning){const m=d.learning;if(typeof m.enabled!=='boolean'||m.validated!==false||typeof m.sha256!=='string'||!count(m.plastic_edges)||m.plastic_edges===0||!count(m.changed_edges)||m.changed_edges>m.plastic_edges||!count(m.damage_events)||!finite(m.delivered_ms)||m.delivered_ms<0||![m.mean_efficacy,m.minimum_efficacy,m.maximum_efficacy,m.mean_absolute_change].every(finite)||!count(m.bound_edges)||!Array.isArray(m.efficacy_histogram)||m.efficacy_histogram.length!==20||!m.efficacy_histogram.every(count)||m.efficacy_histogram.reduce((a,b)=>a+b,0)!==m.plastic_edges||d.reward.plasticity!==m.enabled)throw new Error('Invalid memory telemetry');}
 return d;
}
