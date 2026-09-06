import {env} from 'cloudflare:workers';
import {broadcastProxy} from '@/lib/broadcast-proxy';
export async function GET(request:Request){
 let cache:Cache|undefined;try{cache=(caches as unknown as {default?:Cache}).default;}catch{}
 return broadcastProxy(request,(env as {DOOM_STREAM_ORIGIN?:string}).DOOM_STREAM_ORIGIN,'/index',cache);
}
