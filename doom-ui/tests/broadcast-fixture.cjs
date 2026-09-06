const run='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa';
// Protocol fixtures only; no generated gameplay or neural evidence.
function fixture(sequence=1,time=10000){
 return {schema:1,status:'running',run_id:run,sequence,generated_at_ms:time,condition:'intact',decoder:'bci',frame:'data:image/jpeg;base64,AA==',
 manifest:Object.fromEntries(['neurons','edges','synaptic_contacts','retina_mapped','retina_total','retina_unmapped','uncertain_sign_neurons'].map(k=>[k,1])),
 clocks:{wall_seconds:1,neural_seconds:1,game_seconds:1,speed:1,brain_step_ms:1},game:{episode:1,tick:1,health:100,kills:0,ammo:100,score:0,finished:false},
 action:{turn:0,forward:0,attack:false},readouts:[],total_spikes:1,window_spikes:1,window_ms:100,total_action_ticks:0,
 retina:{uv:[[0,0]],luminance:[.25],full_sample_count:1,display_stride:8},
 raster:{neuron_ids:['1'],bins:[{neural_ms:sequence*100,window_ms:100,counts:[1],population_spikes:1}]},audit:{source_frame_sha256:'fixture'},reward:{plasticity:false},protocol:{phase:'baseline'}};
}
module.exports={run,fixture};
