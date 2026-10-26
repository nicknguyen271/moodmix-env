"""Gymnasium episode: pick each next unplayed song via a Discrete track index."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .rewards import Track, transition_reward

class SongSelectionEnv(gym.Env):
    metadata = {'render_modes':['human']}
    def __init__(self, tracks: list[Track], start_track: int = 0, horizon: int = 3, style: str = 'build'):
        super().__init__()
        if len(tracks)<2 or not 0<=start_track<len(tracks):
            raise ValueError('Supply at least two tracks and a valid start_track')
        if not 1 <= horizon <= len(tracks)-1:
            raise ValueError('Horizon must lie between 1 and len(tracks)-1')
        if style not in ('build','smooth'):
            raise ValueError('style must be build or smooth')
        self.tracks = tuple(tracks)
        self.start_track, self.horizon, self.style = start_track, horizon, style
        n = len(tracks)
        self.action_space = spaces.Discrete(n)
        # Fixed-sized observation: current ID, visited mask, selections remaining, style ID,
        # all candidate BPMs, all candidate key indices/modes, all candidate energies.
        self.observation_space = spaces.Dict({
            'current_track':spaces.Discrete(n),
            'visited':spaces.MultiBinary(n),
            'steps_remaining':spaces.Discrete(horizon+1),
            'style':spaces.Discrete(2),
            'features':spaces.Box(low=0.0,high=1.0,shape=(n,4),dtype=np.float32),
            'action_mask':spaces.MultiBinary(n),
        })
        self.current = start_track
        self.visited = set()
        self.steps_remaining = horizon

    def valid_action_mask(self):
        return np.array([int(i not in self.visited) for i in range(len(self.tracks))],dtype=np.int8)

    def _observation(self):
        features = np.array([[t.bpm/400.0,int(t.camelot[:-1])/12.0,float(t.camelot[-1].upper()=='B'),t.energy] for t in self.tracks], dtype=np.float32)
        visited = np.array([int(i in self.visited) for i in range(len(self.tracks))],dtype=np.int8)
        return {'current_track':self.current,'visited':visited,'steps_remaining':self.steps_remaining,
                'style':int(self.style=='build'),'features':features,'action_mask':self.valid_action_mask()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start = (options or {}).get('start_track',self.start_track)
        if not isinstance(start,(int,np.integer)) or not 0<=start<len(self.tracks):
            raise ValueError('Invalid start_track in reset options')
        self.current=int(start)
        self.visited={self.current}
        self.steps_remaining=self.horizon
        return self._observation(), {'start_track':self.current,'action_mask':self.valid_action_mask()}

    def step(self, action):
        if self.steps_remaining==0:
            raise RuntimeError('Episode finished: call reset()')
        if not self.action_space.contains(action) or int(action) in self.visited:
            raise ValueError(f'Invalid/visited action {action!r}; choose an action where action_mask == 1')
        action=int(action)
        old=self.current
        reward,parts=transition_reward(self.tracks[old],self.tracks[action],self.style)
        self.current=action
        self.visited.add(action)
        self.steps_remaining-=1
        terminated=(self.steps_remaining==0 or len(self.visited)==len(self.tracks))
        info={'from_track':old,'selected_track':action,**parts,'action_mask':self.valid_action_mask()}
        return self._observation(),float(reward),terminated,False,info
