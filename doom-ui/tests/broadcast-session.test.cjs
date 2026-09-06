const test=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {createBroadcastSession,POLL_DEADLINE_MS}=require(path.join(process.env.DOOMFLY_TEST_BUILD,'broadcast-session.js'));
const {broadcastProxy,PROXY_DEADLINE_MS}=require(path.join(process.env.DOOMFLY_TEST_BUILD,'broadcast-proxy.js'));
const {BroadcastPlayer}=require(path.join(process.env.DOOMFLY_TEST_BUILD,'broadcast-player.js'));
const {run,fixture}=require('./broadcast-fixture.cjs');
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const index=(segments=['1788650000'],id=run)=>Response.json({transport:1,status:'running',run_id:id,generated_at_ms:Date.now(),segments,playout_delay_ms:2500,phase:'baseline'});
function packet(sequence=1,id=run){
 const a={...fixture(sequence,Date.now()),run_id:id};const {retina,raster,...shared}=a;
 return Response.json({transport:1,shared:{...shared,retina,raster_ids:raster.neuron_ids},frames:[{...a,luminance:retina.luminance,raster_bin:raster.bins[0]}]});
}
function setup(t){
 t.mock.timers.enable({apis:['setTimeout','Date'],now:1788650000000});
 const original=global.fetch;const frames=[],errors=[];
 const session=createBroadcastSession(batch=>frames.push(...batch),error=>errors.push(error));
 t.after(()=>{session.stop();global.fetch=original;t.mock.timers.reset();});
 return {session,frames,errors};
}
for(const stage of ['headers','body'])test(`a stalled ${stage} cannot prevent timeout and retry`,async t=>{
 const {session,frames,errors}=setup(t);let indexes=0,abandoned;
 global.fetch=async(url,options)=>{
  if(url==='/api/broadcast'){
   indexes++;
   if(indexes===1){abandoned=options.signal;const never=new Promise(()=>{});return stage==='headers'?never:{ok:true,json:()=>never};}
   return index();
  }
  return packet();
 };
 session.resume();await flush();assert.equal(indexes,1);
 t.mock.timers.tick(POLL_DEADLINE_MS);await flush();
 assert.equal(abandoned.aborted,true);assert.match(errors.at(-1),/stalled/);
 t.mock.timers.tick(1000);await flush();
 assert.equal(indexes,2);assert.equal(frames.length,1);assert.equal(errors.at(-1),'');
});
test('hidden tab cancels requests, does no polling, and resumes immediately',async t=>{
 const {session,frames}=setup(t);let calls=0,abandoned;
 global.fetch=async(url,options)=>{calls++;if(calls===1){abandoned=options.signal;return new Promise(()=>{});}return url==='/api/broadcast'?index():packet();};
 session.resume();await flush();session.pause();assert.equal(abandoned.aborted,true);
 t.mock.timers.tick(60000);await flush();assert.equal(calls,1);
 session.resume();await flush();assert.equal(frames.length,1);assert.equal(calls,3);
});
test('returning to a page replaces a stuck generation without waiting for its timeout',async t=>{
 const {session,frames}=setup(t);let calls=0;
 global.fetch=async(url)=>{calls++;return calls===1?new Promise(()=>{}):url==='/api/broadcast'?index():packet();};
 session.resume();await flush();session.resume();await flush();
 assert.equal(frames.length,1);assert.equal(calls,3);
 t.mock.timers.tick(750);await flush();assert.equal(calls,4,'one poll loop remains');
});
test('late packets from an abandoned run cannot replace the new run',async t=>{
 const {session,frames}=setup(t);let indexes=0,release;
 const newer='bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb';
 global.fetch=async url=>{
  if(url==='/api/broadcast')return index(['1788650000'],++indexes===1?run:newer);
  if(url.includes(run))return new Promise(resolve=>{release=resolve;});
  return packet(1,newer);
 };
 session.resume();await flush();session.pause();session.resume();await flush();
 assert.deepEqual(frames.map(f=>f.run_id),[newer]);
 release(packet(99,run));await flush();
 assert.deepEqual(frames.map(f=>f.run_id),[newer]);
});
test('one hung batch does not hold up other available frames or future polls',async t=>{
 const {session,frames}=setup(t);let indexes=0;
 global.fetch=async url=>{
  if(url==='/api/broadcast')return index(++indexes===1?['1788650000','1788650001']:['1788650001','1788650002']);
  if(url.endsWith('/1788650000'))return new Promise(()=>{});
  return packet(url.endsWith('/1788650001')?2:3);
 };
 session.resume();await flush();assert.deepEqual(frames.map(f=>f.sequence),[2]);
 t.mock.timers.tick(POLL_DEADLINE_MS);await flush();t.mock.timers.tick(750);await flush();
 assert.deepEqual(frames.map(f=>f.sequence),[2,3]);
});
test('unmount stops retries and ignores a late response',async t=>{
 const {session,frames,errors}=setup(t);let calls=0,release;
 global.fetch=async()=>{calls++;return new Promise(resolve=>{release=resolve;});};
 session.resume();await flush();session.stop();release(index());await flush();
 t.mock.timers.tick(60000);await flush();session.resume();await flush();
 assert.equal(calls,1);assert.equal(frames.length,0);assert.equal(errors.length,0);
});
test('a stale index remains offline and a manual reconnect recovers',async t=>{
 const {session,frames,errors}=setup(t);let calls=0;
 global.fetch=async url=>{
  if(url==='/api/broadcast'&&++calls===1){const d=await index().json();d.generated_at_ms-=20000;return Response.json(d);}
  return url==='/api/broadcast'?index():packet();
 };
 session.resume();await flush();assert.equal(frames.length,0);assert.match(errors.at(-1),/fresh frames/);
 session.resume();await flush();assert.equal(frames.length,1);assert.equal(errors.at(-1),'');
});
test('proxy deadline covers a response body that never finishes',async t=>{
 setup(t);let upstreamSignal;
 global.fetch=async(_url,{signal})=>{upstreamSignal=signal;return new Response(new ReadableStream({start(controller){controller.enqueue(new TextEncoder().encode('{'));}}));};
 const response=broadcastProxy(new Request('https://site.test/api/broadcast'),'https://brain.test','/index');
 await flush();t.mock.timers.tick(PROXY_DEADLINE_MS);await flush();
 assert.equal((await response).status,503);assert.equal(upstreamSignal.aborted,true);
});
test('resuming drops queued old frames and retains already displayed neural history',()=>{
 const player=new BroadcastPlayer();player.ingest([fixture(1,10000),fixture(2,11000)]);
 assert.equal(player.take(12500).sequence,1);player.discardPending();assert.equal(player.take(20000),null);
 player.ingest([fixture(3,20000)]);const next=player.take(22500);
 assert.equal(next.sequence,3);assert.deepEqual(next.raster.bins.map(b=>b.neural_ms),[100,300]);
});
