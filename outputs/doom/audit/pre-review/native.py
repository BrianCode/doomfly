"""Native SIMD implementation of the fixed-step model; no changed graph."""
import ctypes as C
from pathlib import Path
import math,time,sys
import numpy as np
from doom.engine import Brain
ROOT=Path(__file__).resolve().parents[1]
_lib=C.CDLL(str(ROOT/'outputs/doom'/('libneural.dylib' if sys.platform=='darwin' else 'libneural.so')))
_f=_lib.neural_advance
_f.argtypes=[C.c_int]+[C.c_void_p]*11+[C.c_int,C.c_float]+[C.c_void_p]*5
_f.restype=None
class NativeBrain(Brain):
    def __init__(self,path,dt=.1):
        super().__init__(path,dt)
        self.previous_drive=np.zeros(self.n,dtype=np.float32)
        self.last=np.full(self.n,-1,dtype=np.int64)
    def step(self,luminance,duration_ms,sugar=False,lamina_bias=12.0):
        if len(luminance)!=len(self.retina) or not np.all(np.isfinite(luminance)):raise ValueError('Invalid retinal input')
        steps=int(round(duration_ms/self.dt))
        if steps<1:raise ValueError('Duration too short')
        self.luminance+=(1-math.exp(-steps*self.dt/10))*(np.clip(luminance,0,1)-self.luminance)
        self.drive.fill(0);self.drive[self.lamina]=lamina_bias;self.drive[self.retina]=30*self.luminance/(.02+self.luminance)
        if sugar:self.drive[self.sugar]=30
        self.counts.fill(0);clock=np.asarray([self.cursor],dtype=np.int64)
        arrays=[self.ptr,self.post,self.weight,self.v,self.g,self.refractory,self.drive,self.previous_drive,self.queue,self.queue_count,clock]
        start=time.perf_counter()
        _f(self.n,*[x.ctypes.data for x in arrays],steps,self.dt,*[x.ctypes.data for x in [self.counts,self.active,self.active_flag,self.nactive,self.last]])
        wall=time.perf_counter()-start
        self.cursor=int(clock[0]);self.total_spikes+=int(self.counts.sum());self.sim_ms+=steps*self.dt
        return self.counts.copy(),wall
