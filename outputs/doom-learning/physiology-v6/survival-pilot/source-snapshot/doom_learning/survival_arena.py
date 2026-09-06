"""A small, real Doom hazard task. No controller or neural policy lives here.

UDMF sector damage is evaluated by ViZDoom itself. Positions are available for
the experiment observer only; sensory input remains the RGB screen buffer.
The generated map and original flat textures are reproducible from their seed.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import vizdoom as vzd


def lumps(path):
    data=Path(path).read_bytes();_,n,offset=struct.unpack_from('<4sii',data)
    return {name.rstrip(b'\0').decode('ascii'):data[start:start+size]
            for start,size,name in (struct.unpack_from('<ii8s',data,offset+16*i) for i in range(n))}


def write_wad(path,entries):
    body=bytearray();directory=bytearray()
    for name,data in entries:
        directory+=struct.pack('<ii8s',12+len(body),len(data),name.encode('ascii'))
        body+=data
    Path(path).write_bytes(struct.pack('<4sii',b'PWAD',len(entries),12+len(body))+body+directory)


def build_map(path,seed=41031,*,hazard_left=True,angle=None,damage=20):
    """Seed changes spawn position/facing, never the behavior of the controller."""
    rng=np.random.default_rng(seed)
    iwad=Path(vzd.__file__).parent/'freedoom2.wad'
    palette=np.frombuffer(lumps(iwad)['PLAYPAL'][:768],dtype=np.uint8).reshape(256,3)
    def flat(color):
        p=int(np.argmin(np.square(palette.astype(float)-color).sum(axis=1)))
        return bytes([p])*4096,{'palette_index':p,'actual_srgb':palette[p].tolist()}
    blue,blue_info=flat([0,0,255]);gray,gray_info=flat([96,96,96])
    vertices=[(-256,-256),(-256,256),(0,256),(256,256),(256,-256),(0,-256)]
    text=['namespace = "ZDoom";']
    for x,y in vertices:text.append(f'vertex {{ x={x}.0; y={y}.0; }}')
    for sector in range(2):
        hazard=(sector==0)==hazard_left
        text.append('sector { heightfloor=0; heightceiling=128; texturefloor="%s"; '
                    'textureceiling="CEIL1_1"; lightlevel=255; damageamount=%s; '
                    'damageinterval=32; leakiness=256; }'%('BLUEHAZ' if hazard else 'SAFEGRAY',damage if hazard else 0))
    walls=[(0,1,0),(1,2,0),(2,3,1),(3,4,1),(4,5,1),(5,0,0)]
    for i,(a,b,sector) in enumerate(walls):
        text.append(f'sidedef {{ sector={sector}; texturemiddle="STARTAN3"; }}')
        text.append(f'linedef {{ v1={a}; v2={b}; sidefront={i}; blocking=true; }}')
    text+=['sidedef { sector=0; }','sidedef { sector=1; }',
           'linedef { v1=2; v2=5; sidefront=6; sideback=7; twosided=true; }']
    x=float(rng.uniform(-180,-100))*(1 if hazard_left else -1)
    y=float(rng.uniform(-140,140));heading=int(rng.integers(0,360)) if angle is None else int(angle)
    text.append(f'thing {{ x={x:.4f}; y={y:.4f}; angle={heading}; type=1; '
                'skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; single=true; }')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    write_wad(path,[('MAP01',b''),('TEXTMAP','\n'.join(text).encode()),('ENDMAP',b''),
                    ('F_START',b''),('BLUEHAZ',blue),('SAFEGRAY',gray),('F_END',b'')])
    metadata={'task':'blue-floor-survival-v1','seed':seed,'hazard_left':hazard_left,
        'spawn':[x,y,heading],'damage':damage,'damage_interval_tics':32,
        'textures':{'hazard':blue_info,'safe':gray_info},
        'wad_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'iwad_sha256':hashlib.sha256(iwad.read_bytes()).hexdigest(),
        'engine':vzd.__version__,'engine_hz':35,
        'source':'https://github.com/ZDoom/gzdoom/blob/master/specs/udmf_zdoom.txt',
        'limits':'Engine hazard, not a biological environment. A 60-second cap is censored survival, not an escape or learning claim.'}
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


class SurvivalArena:
    def __init__(self,path,seed=41031,*,seconds=60,hazard_left=True,angle=None):
        self.assets=build_map(path,seed,hazard_left=hazard_left,angle=angle)
        self.game=g=vzd.DoomGame()
        g.set_doom_game_path(str(Path(vzd.__file__).parent/'freedoom2.wad'))
        g.set_doom_scenario_path(str(Path(path).resolve()));g.set_doom_map('map01')
        g.set_window_visible(False);g.set_sound_enabled(False)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        g.set_mode(vzd.Mode.PLAYER);g.set_render_hud(False)
        g.set_depth_buffer_enabled(False);g.set_labels_buffer_enabled(False)
        g.set_automap_buffer_enabled(False);g.set_objects_info_enabled(False)
        g.set_sectors_info_enabled(False)
        g.set_available_buttons([vzd.Button.TURN_LEFT_RIGHT_DELTA,vzd.Button.MOVE_FORWARD_BACKWARD_DELTA,vzd.Button.ATTACK])
        g.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA,6)
        g.set_button_max_value(vzd.Button.MOVE_FORWARD_BACKWARD_DELTA,20)
        g.set_available_game_variables([vzd.GameVariable.HEALTH,vzd.GameVariable.KILLCOUNT,vzd.GameVariable.AMMO2])
        g.set_episode_timeout(round(35*seconds));g.set_episode_start_time(0)
        g.set_seed(seed);g.init();g.new_episode();self.tick=0

    def pixels(self):
        state=self.game.get_state()
        if state is None:raise RuntimeError('Episode finished')
        return state.screen_buffer.copy()

    def act(self,action):
        r=self.game.make_action([action['turn'],action['forward'],int(action['attack'])],1)
        self.tick+=1
        return float(r)

    def observation(self):
        g=self.game
        return {'tick':self.tick,'finished':g.is_episode_finished(),'dead':g.is_player_dead(),
                'health':float(g.get_game_variable(vzd.GameVariable.HEALTH)),
                'kills':int(g.get_game_variable(vzd.GameVariable.KILLCOUNT)),
                'ammo':int(g.get_game_variable(vzd.GameVariable.AMMO2))}

    def observer_position(self):
        return [float(self.game.get_game_variable(v)) for v in [vzd.GameVariable.POSITION_X,vzd.GameVariable.POSITION_Y]]

    def close(self):self.game.close()


def check(out):
    """Explicit programmed actions test the map. These are NOT fly behavior."""
    from PIL import Image
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for name,forward in [('stationary',0),('cross_to_safe',20)]:
        a=SurvivalArena(out/f'{name}.wad',seconds=12,angle=0)
        Image.fromarray(a.pixels()).save(out/f'{name}.png')
        samples=[]
        while not a.observation()['finished']:
            # Only this environment test supplies a programmed control.
            a.act({'turn':0,'forward':forward,'attack':False})
            if a.tick%16==0 or a.observation()['finished']:
                samples.append({**a.observation(),'position':a.observer_position()})
        rows.append({'test':name,'programmed_environment_test':True,'samples':samples,'assets':a.assets});a.close()
    stationary,cross=rows
    passed=stationary['samples'][-1]['dead'] and not cross['samples'][-1]['dead'] and cross['samples'][-1]['position'][0]>0
    report={'environment_test_passed':bool(passed),'fly_behavior_tested':False,'rows':rows}
    (out/'environment-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':bool(passed),'final':[r['samples'][-1] for r in rows]}))
    if not passed:raise AssertionError('Doom hazard/safe-sector test failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/doom-learning/survival-arena/check')
    a=p.parse_args();check(a.out)
