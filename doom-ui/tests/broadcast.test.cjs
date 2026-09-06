const test=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const {BroadcastPlayer,parseSegment,parseBroadcastIndex}=require(path.join(process.env.DOOMFLY_TEST_BUILD,'broadcast-player.js'));
const {broadcastProxy}=require(path.join(process.env.DOOMFLY_TEST_BUILD,'broadcast-proxy.js'));
const {parseLive}=require(path.join(process.env.DOOMFLY_TEST_BUILD,'live.js'));
const {run,fixture}=require('./broadcast-fixture.cjs');
test('actual timestamps determine playback; repeated packets never invent frames',()=>{
 const p=new BroadcastPlayer();const a=fixture(),b=fixture(2,10125),c=fixture(3,10300);
 p.ingest([a,b,c]);assert.equal(p.take(12500).sequence,1);
 assert.equal(p.take(12550),null);p.ingest([a,b,c]);assert.equal(p.take(12625).sequence,2);
 assert.equal(p.take(12800).sequence,3);assert.equal(p.take(12900),null);
});
test('gaps remain gaps; new runs clear history and sequence',()=>{
 const p=new BroadcastPlayer();p.ingest([fixture(),fixture(4,12000)]);p.take(12500);
 assert.equal(p.take(14000),null);assert.deepEqual(p.take(14500).raster.bins.map(b=>b.neural_ms),[100,400]);
 p.ingest([{...fixture(1,15000),run_id:'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'}]);
 assert.equal(p.take(17500).raster.bins.length,1);
});
test('packet reconstruction preserves exact action, image and neuronal bin',()=>{
 const a=fixture();const {retina,raster,...shared}=a;
 const data=parseSegment({transport:1,shared:{...shared,retina,raster_ids:raster.neuron_ids},frames:[{...a,luminance:retina.luminance,raster_bin:raster.bins[0]}]},run)[0];
 assert.equal(data.frame,a.frame);assert.deepEqual(data.action,a.action);assert.deepEqual(data.raster,a.raster);
 assert.throws(()=>parseSegment({transport:1,shared:{run_id:'wrong'},frames:[]},run));
 assert.throws(()=>parseBroadcastIndex({transport:1,status:'running',run_id:run,generated_at_ms:1,segments:['../../bad'],playout_delay_ms:2500,phase:'baseline'}));
});
test('completed bytes are cached across viewers; stale feed stays offline',async()=>{
 const original=global.fetch;let calls=0;const items=new Map();
 const cache={async match(request){return items.get(request.url)?.clone();},async put(request,response){items.set(request.url,response);}};
 global.fetch=async()=>{calls++;return Response.json({transport:1,status:'running',generated_at_ms:Date.now()});};
 try{
  for(let i=0;i<25;i++)assert.equal((await broadcastProxy(new Request('https://site.test/api/broadcast'),'https://brain.test','/index',cache)).status,200);
  assert.equal(calls,1);
  items.clear();global.fetch=async()=>Response.json({transport:1,status:'running',generated_at_ms:Date.now()-10000});
  assert.equal((await broadcastProxy(new Request('https://site.test/api/broadcast'),'https://brain.test','/index',cache)).status,503);
 }finally{global.fetch=original;}
});
test('canceling one viewer never strands another request in a shared promise',async()=>{
 const original=global.fetch;let calls=0;const first=new AbortController();let firstSignal;
 global.fetch=async(_url,options)=>{calls++;if(calls===1){firstSignal=options.signal;return new Promise(()=>{});}return Response.json({transport:1,status:'running',generated_at_ms:Date.now()});};
 try{
  const abandoned=broadcastProxy(new Request('https://site.test/api/broadcast',{signal:first.signal}),'https://brain.test','/index');
  await new Promise(resolve=>setImmediate(resolve));
  first.abort();
  assert.equal((await broadcastProxy(new Request('https://site.test/api/broadcast'),'https://brain.test','/index')).status,200);
  assert.equal((await abandoned).status,503);assert.equal(firstSignal.aborted,true);assert.equal(calls,2);
 }finally{global.fetch=original;}
});
test('immutable segments get CDN headers; invalid paths never reach origin',async()=>{
 const original=global.fetch;let calls=0;
 global.fetch=async()=>{calls++;return Response.json({transport:1,frames:[]});};
 try{
  const response=await broadcastProxy(new Request('https://site.test/api/broadcast'),'https://brain.test',`/segments/${run}/1788650000`);
  assert.match(response.headers.get('Cache-Control'),/immutable/);
  assert.equal((await broadcastProxy(new Request('https://site.test'),'https://brain.test','/secrets')).status,404);
  assert.equal(calls,1);
 }finally{global.fetch=original;}
});
test('training labels require real matching memory telemetry',()=>{
 const a=fixture();a.protocol.phase='training';a.reward.plasticity=true;
 assert.throws(()=>parseLive(a),/training evidence/);
 a.learning={enabled:true,validated:false,model:'fixture',sha256:'fixture',plastic_edges:1,changed_edges:0,mean_efficacy:1,minimum_efficacy:1,maximum_efficacy:1,mean_absolute_change:0,bound_edges:0,damage_events:0,delivered_ms:0,efficacy_histogram:[1,...Array(19).fill(0)]};
 assert.equal(parseLive(a).learning.enabled,true);
 assert.equal(parseBroadcastIndex({transport:1,status:'running',run_id:run,generated_at_ms:1,segments:[],playout_delay_ms:2500,phase:'training'}).phase,'training');
 a.learning.efficacy_histogram[0]=2;assert.throws(()=>parseLive(a),/memory telemetry/);
});
