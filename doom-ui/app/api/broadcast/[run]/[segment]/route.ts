import {env} from 'cloudflare:workers';
import {broadcastProxy} from '@/lib/broadcast-proxy';
export async function GET(request:Request,context:{params:Promise<{run:string;segment:string}>}){
 const {run,segment}=await context.params;
 let cache:Cache|undefined;try{cache=(caches as unknown as {default?:Cache}).default;}catch{}
 return broadcastProxy(request,(env as {DOOM_STREAM_ORIGIN?:string}).DOOM_STREAM_ORIGIN,`/segments/${run}/${segment}`,cache);
}
