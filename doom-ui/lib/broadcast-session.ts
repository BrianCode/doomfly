import type {Live} from './live';
import {parseBroadcastIndex,parseSegment} from './broadcast-player';
import {withDeadline} from './request-deadline';

export const POLL_DEADLINE_MS=6000;

// Network lifetime is independent of React renders. A resumed tab gets a new
// request generation immediately; late results from its old generation are ignored.
export function createBroadcastSession(onFrames:(frames:Live[])=>void,onError:(message:string)=>void){
 const fetched=new Set<string>();
 let timer:ReturnType<typeof setTimeout>|undefined;
 let active:AbortController|null=null;
 let paused=true,stopped=false,failures=0,run='';
 const poll=async()=>{
  if(paused||stopped||active)return;
  const controller=new AbortController();active=controller;
  const current=()=>!paused&&!stopped&&active===controller&&!controller.signal.aborted;
  let received=false;
  try{
   await withDeadline(async signal=>{
    const response=await fetch('/api/broadcast',{signal,cache:'no-store'});
    if(!response.ok)throw new Error('Waiting for the simulation host');
    const index=parseBroadcastIndex(await response.json());
    if(!current()||signal.aborted)return;
    if(Math.abs(Date.now()-index.generated_at_ms)>5000)throw new Error('Waiting for fresh frames');
    if(run!==index.run_id){run=index.run_id;fetched.clear();}
    const needed=index.segments.filter(segment=>!fetched.has(`${run}/${segment}`));
    const results=await Promise.allSettled(needed.map(async segment=>{
     const key=`${index.run_id}/${segment}`;
     const r=await fetch(`/api/broadcast/${key}`,{signal});
     if(!r.ok)throw new Error('Waiting for the next frame batch');
     const frames=parseSegment(await r.json(),index.run_id);
     if(!current()||signal.aborted)return;
     // One unavailable batch cannot prevent other completed batches playing.
     onFrames(frames);fetched.add(key);received=true;failures=0;onError('');
     if(fetched.size>40){const retained=[...fetched].slice(-20);fetched.clear();retained.forEach(x=>fetched.add(x));}
    }));
    if(!current()||signal.aborted)return;
    const failed=results.find(r=>r.status==='rejected');
    if(!received&&failed?.status==='rejected')throw failed.reason;
    failures=0;onError('');
   },controller.signal,POLL_DEADLINE_MS);
  }catch(error){
   if(current()&&!received){failures++;onError(error instanceof DOMException&&error.name==='TimeoutError'?'Connection stalled. Reconnecting…':error instanceof Error?error.message:'Broadcast unavailable');}
  }finally{
   if(active===controller){
    active=null;controller.abort();
    if(!paused&&!stopped)timer=setTimeout(poll,failures?Math.min(5000,1000*failures):750);
   }
  }
 };
 const pause=()=>{
  paused=true;clearTimeout(timer);timer=undefined;
  const previous=active;active=null;previous?.abort();
 };
 return {
  pause,
  resume(){if(stopped)return;pause();paused=false;failures=0;void poll();},
  stop(){pause();stopped=true;fetched.clear();},
 };
}
