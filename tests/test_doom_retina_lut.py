"""Table-driven receptor sampling must be bit-identical to the reference functions."""
import numpy as np
from doom.game import retinal_samples
from doom.retina_lut import RetinaSampler, ChannelSampler

def test_retina_sampler_is_bit_identical_to_retinal_samples():
    rng=np.random.default_rng(5)
    for shape in [(48,64),(480,640)]:
        uv=rng.uniform(0,1,(500,2)).astype(np.float32); uv[:4]=[[0,0],[1,1],[0,1],[1,0]]
        sampler=RetinaSampler(uv,shape)
        for _ in range(3):
            frame=rng.integers(0,256,(*shape,3),dtype=np.uint8)
            np.testing.assert_array_equal(sampler(frame),retinal_samples(frame,uv))
    white=np.full((48,64,3),255,np.uint8); assert np.allclose(RetinaSampler(uv,(48,64))(white),1,atol=1e-6)

def test_channel_sampler_matches_rgb_step_linearisation():
    rng=np.random.default_rng(6); h,w=48,64
    uv=rng.uniform(0,1,(300,2)).astype(np.float32); channel=rng.integers(1,3,300).astype(np.int32)
    frame=rng.integers(0,256,(h,w,3),dtype=np.uint8)
    x=np.minimum((uv[:,0]*(w-1)).astype(int),w-1); y=np.minimum((uv[:,1]*(h-1)).astype(int),h-1)
    values=frame[y,x,channel].astype(np.float32)/255
    values=np.where(values<=.04045,values/12.92,((values+.055)/1.055)**2.4)
    np.testing.assert_array_equal(ChannelSampler(uv,channel,(h,w))(frame),values)
