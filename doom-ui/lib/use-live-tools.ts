'use client';
import {useEffect,useRef} from 'react';
import {flushSync} from 'react-dom';
type Context={registerTool(tool:{name:string;description:string;inputSchema:object;annotations:object;execute(input:unknown):unknown},options:{signal:AbortSignal}):unknown};
export function useLiveTools(read:()=>unknown,apply:(retina:boolean,activity:boolean)=>void){
 const ref=useRef({read,apply});ref.current={read,apply};
 useEffect(()=>{const context=(document as Document & {modelContext?:Context}).modelContext;if(!context?.registerTool)return;
 const life=new AbortController();const obj=(x:unknown)=>{if(!x||typeof x!=='object'||Array.isArray(x))throw new Error('Expected an object');return x as Record<string,unknown>};
 const tools=[{name:'get_live_fly_doom',description:'Read the current live broadcast status, actual neural controls, clocks, and model limitations. This does not advance or control the game.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true,untrustedContentHint:false},execute(x:unknown){if(Object.keys(obj(x)).length)throw new Error('Provide an empty object');return ref.current.read();}},
 {name:'set_fly_doom_overlays',description:'Show or hide the retinal-input and neural-activity panels for this viewer. Does not alter the shared experiment or game controls.',inputSchema:{type:'object',properties:{retina:{type:'boolean'},activity:{type:'boolean'}},required:['retina','activity'],additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:false},execute(x:unknown){const v=obj(x);if(Object.keys(v).length!==2||typeof v.retina!=='boolean'||typeof v.activity!=='boolean')throw new Error('Provide retina and activity as booleans');flushSync(()=>ref.current.apply(v.retina as boolean,v.activity as boolean));return ref.current.read();}}];
 for(const tool of tools){try{void Promise.resolve(context.registerTool(tool,{signal:life.signal})).catch(()=>{})}catch{}}
 return()=>life.abort();},[]);
}
