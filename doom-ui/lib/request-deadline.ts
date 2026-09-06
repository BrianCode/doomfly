// Bound the whole operation, including reading its body. An abandoned request
// must release its caller even if the underlying transport never settles.
export async function withDeadline<T>(work:(signal:AbortSignal)=>Promise<T>,parent:AbortSignal,timeoutMs:number):Promise<T>{
 const controller=new AbortController();
 let timer:ReturnType<typeof setTimeout>|undefined;
 let cancel=()=>{};
 const interrupted=new Promise<never>((_,reject)=>{
  cancel=()=>{const reason=parent.reason??new DOMException('Request canceled','AbortError');controller.abort(reason);reject(reason);};
  if(parent.aborted){cancel();return;}
  parent.addEventListener('abort',cancel,{once:true});
  timer=setTimeout(()=>{const reason=new DOMException('Broadcast request timed out','TimeoutError');controller.abort(reason);reject(reason);},timeoutMs);
 });
 try{
  return await Promise.race([interrupted,Promise.resolve().then(()=>{controller.signal.throwIfAborted();return work(controller.signal);})]);
 }finally{clearTimeout(timer);parent.removeEventListener('abort',cancel);}
}
