"""Doom boundary: the neural input path receives pixels, never object state."""
from pathlib import Path
import hashlib
import numpy as np
import vizdoom as vzd

class Game:
    def __init__(self,seed=41027,scenario='defend_the_center'):
        self.game=vzd.DoomGame()
        cfg=Path(vzd.scenarios_path)/(scenario+'.cfg')
        wad=Path(vzd.scenarios_path)/(scenario+'.wad')
        iwad=Path(vzd.__file__).parent/'freedoom2.wad'
        self.game.load_config(str(cfg))
        self.game.set_doom_game_path(str(iwad))
        self.assets={'vizdoom_version':vzd.__version__,'scenario':scenario,
          'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [cfg,wad,iwad]}}
        self.game.set_window_visible(False);self.game.set_sound_enabled(False)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.set_depth_buffer_enabled(False);self.game.set_labels_buffer_enabled(False)
        self.game.set_automap_buffer_enabled(False);self.game.set_objects_info_enabled(False)
        self.game.set_sectors_info_enabled(False)
        self.game.set_available_buttons([vzd.Button.TURN_LEFT_RIGHT_DELTA,vzd.Button.MOVE_FORWARD_BACKWARD_DELTA,vzd.Button.ATTACK])
        self.game.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA,6)
        self.game.set_button_max_value(vzd.Button.MOVE_FORWARD_BACKWARD_DELTA,20)
        self.game.clear_available_game_variables()
        # These are observer/reinforcement outputs only. Never fed to controls.
        self.game.add_available_game_variable(vzd.GameVariable.HEALTH)
        self.game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
        self.game.add_available_game_variable(vzd.GameVariable.AMMO2)
        self.game.set_episode_timeout(35*60)
        self.game.set_seed(seed);self.game.init()
        self.episode=0;self.tick=0;self.episodes=[];self.new_episode()
    def new_episode(self):
        self.game.new_episode();self.episode+=1;self.tick=0
    def pixels(self):
        state=self.game.get_state()
        if state is None:raise RuntimeError('Episode finished; reset is required')
        return state.screen_buffer.copy()
    def act(self,action):
        # The adapter is the only caller of make_action. No human keystrokes.
        reward=self.game.make_action([action['turn'],action['forward'],int(action['attack'])],1)
        self.tick+=1
        return float(reward)
    def observation(self):
        g=self.game
        return {'episode':self.episode,'tick':self.tick,'finished':g.is_episode_finished(),
          'health':int(g.get_game_variable(vzd.GameVariable.HEALTH)),
          'kills':int(g.get_game_variable(vzd.GameVariable.KILLCOUNT)),
          'ammo':int(g.get_game_variable(vzd.GameVariable.AMMO2)),
          'score':float(g.get_total_reward())}
    def close(self):self.game.close()

def retinal_samples(rgb,uv):
    """Bilinear luminance at receptor samples only. No scene interpretation."""
    h,w=rgb.shape[:2];x=uv[:,0]*(w-1);y=uv[:,1]*(h-1)
    x0=x.astype(int);y0=y.astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    dx=x-x0;dy=y-y0
    def linear_luma(pixels):
        p=pixels.astype(np.float32)/255
        p=np.where(p<=.04045,p/12.92,((p+.055)/1.055)**2.4)
        return p@np.asarray([.2126,.7152,.0722],dtype=np.float32)
    return ((1-dx)*(1-dy)*linear_luma(rgb[y0,x0])+dx*(1-dy)*linear_luma(rgb[y0,x1])+(1-dx)*dy*linear_luma(rgb[y1,x0])+dx*dy*linear_luma(rgb[y1,x1])).astype(np.float32)
