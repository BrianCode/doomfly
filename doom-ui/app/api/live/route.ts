import { env } from 'cloudflare:workers';
import {parseLive} from '@/lib/live';
export async function GET(request:Request){
 const origin=(env as {DOOM_STREAM_ORIGIN?:string}).DOOM_STREAM_ORIGIN;
 const offline=(message:string)=>Response.json({status:'offline',message},{status:503,headers:{'Cache-Control':'no-store'}});
 if(!origin)return offline('The broadcaster is not connected.');
 try{
  const url=new URL(origin);
  if(url.protocol!=='https:' && !(url.protocol==='http:' && ['127.0.0.1','localhost'].includes(url.hostname)))throw new Error('Invalid configured origin');
  // Short edge cache fans out one shared broadcast without asking the neural
  // machine to reserialize a frame for every spectator. Browser remains fresh.
  let cache:Cache|undefined;
  try{cache=(caches as unknown as {default?:Cache}).default;}catch{cache=undefined;}
  const cacheKey=new Request(new URL('/__live_frame_cache',request.url));
  let cached:Response|undefined;
  try{cached=cache?await cache.match(cacheKey):undefined;}catch{cache=undefined;}
  let data;
  if(cached){data=parseLive(await cached.json());}
  else{
   const upstream=await fetch(new URL('/state',url),{signal:AbortSignal.timeout(4000),headers:{Accept:'application/json'}});
   if(!upstream.ok)throw new Error('Broadcaster unavailable');
   const text=await upstream.text();if(text.length>1_500_000)throw new Error('Invalid frame size');
   data=parseLive(JSON.parse(text));
   if(cache && Date.now()-data.generated_at_ms<3000){try{await cache.put(cacheKey,Response.json(data,{headers:{'Cache-Control':'public, max-age=1'}}));}catch{/* Some hosted isolates do not provide Cache API. Live fetch still works. */}}
  }
  if(Date.now()-data.generated_at_ms>5000)throw new Error('Stale broadcast');
  return Response.json(data,{headers:{'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}});
 }catch(error){console.error('Doom live proxy:',error instanceof Error?error.message:'unknown error');return offline('The broadcaster is offline. No live frames are available.')}
}
