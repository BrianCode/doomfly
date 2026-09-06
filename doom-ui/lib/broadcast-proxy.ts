import {withDeadline} from './request-deadline';

// Cache completed bytes, never in-flight promises: Worker I/O belongs to the
// request that created it. Another viewer must survive that request canceling.
type EdgeCache={match(request:Request):Promise<Response|undefined>;put(request:Request,response:Response):Promise<unknown>};
const maximumBytes=3_000_000;
export const PROXY_DEADLINE_MS=4000;

export async function broadcastProxy(request:Request,origin:string|undefined,path:string,cache?:EdgeCache):Promise<Response>{
 const offline=()=>Response.json({status:'offline',message:'Waiting for the simulation host.'},{status:503,headers:{'Cache-Control':'no-store'}});
 if(!origin)return offline();
 const index=path==='/index';
 if(!index&&!/^\/segments\/[a-f0-9-]{36}\/[0-9]{10,14}$/.test(path))return new Response('Not found',{status:404});
 try{
  return await withDeadline(async signal=>{
   const url=new URL(origin);
   if(url.protocol!=='https:'&&!(url.protocol==='http:'&&['localhost','127.0.0.1'].includes(url.hostname)))return offline();
   const key=new Request(new URL(`/__doom_broadcast${path}?origin=${encodeURIComponent(url.origin)}`,request.url));
   let cached:Response|undefined;
   try{cached=await cache?.match(key);}catch{cache=undefined;}
   signal.throwIfAborted();
   let body:string;
   if(cached)body=await cached.text();
   else{
    const response=await fetch(new URL(path,url),{signal,headers:{Accept:'application/json'}});
    if(!response.ok||!response.body)throw new Error('Unavailable segment');
    const reader=response.body.getReader();const decoder=new TextDecoder();let bytes=0,text='';
    try{
     while(true){const chunk=await reader.read();if(chunk.done)break;bytes+=chunk.value.byteLength;if(bytes>maximumBytes)throw new Error('Oversized segment');text+=decoder.decode(chunk.value,{stream:true});}
     text+=decoder.decode();
    }finally{void reader.cancel().catch(()=>{});}
    signal.throwIfAborted();
    const data=JSON.parse(text);
    if(data.transport!==1||(index?data.status!=='running':!Array.isArray(data.frames)))throw new Error('Invalid broadcast');
    if(index&&(typeof data.generated_at_ms!=='number'||Math.abs(Date.now()-data.generated_at_ms)>5000))throw new Error('Stale broadcast');
    try{await cache?.put(key,new Response(text,{headers:{'Content-Type':'application/json','Cache-Control':index?'public, max-age=1':'public, max-age=31536000, immutable'}}));}catch{/* A cache failure must not replace usable origin bytes. */}
    body=text;
   }
   signal.throwIfAborted();
   if(index){const data=JSON.parse(body);if(typeof data.generated_at_ms!=='number'||Math.abs(Date.now()-data.generated_at_ms)>5000)return offline();}
   return new Response(body,{headers:{'Content-Type':'application/json','X-Content-Type-Options':'nosniff',
    'Cache-Control':index?'public, max-age=0, must-revalidate':'public, max-age=31536000, immutable',
    'CDN-Cache-Control':index?'public, s-maxage=1':'public, s-maxage=31536000, immutable',
    'Vercel-CDN-Cache-Control':index?'public, s-maxage=1':'public, s-maxage=31536000, immutable'}});
  },request.signal,PROXY_DEADLINE_MS);
 }catch{return offline();}
}
