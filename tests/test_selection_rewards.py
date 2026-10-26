import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moodmix_env.song_selection.music_theory import harmonic_score,circle_distance
from moodmix_env.song_selection.rewards import Track,bpm_score,energy_score,transition_reward
from moodmix_env.song_selection.baselines import optimal_route,greedy_route,evaluate_route

class TestRewards(unittest.TestCase):
    def test_harmonic(self):
        for target,expected in [('8A',1),('8B',.95),('9A',.85),('7A',.85),('10A',.5),('2A',.1)]:
            self.assertAlmostEqual(harmonic_score('8A',target),expected)
        self.assertAlmostEqual(harmonic_score('12A','1A'),.85)
        self.assertEqual(circle_distance('12A','1A'),1)
    def test_bpm(self):
        for value,expected in [(124,1),(126,5/6),(128,2/3),(132,1/3),(140,0)]:
            self.assertAlmostEqual(bpm_score(124,value),expected)
    def test_energy(self):
        self.assertAlmostEqual(energy_score(.6,.7,'build'),1)
        self.assertAlmostEqual(energy_score(.6,.7,'smooth'),2/3)
    def test_examples(self):
        src=Track('a',124,'8A',.60)
        dest=[Track('b',126,'8A',.70),Track('c',128,'9A',.75),Track('d',124,'8B',.60),Track('e',140,'2A',.90)]
        expected=[.94166666667,.7825,.91083333333,.11166666667]
        for t,x in zip(dest,expected):self.assertAlmostEqual(transition_reward(src,t,'build')[0],x)
    def test_bounds(self):
        tracks=[Track('a',120,'8A',.1),Track('b',145,'2B',1),Track('c',121,'7A',.2)]
        for a in tracks:
            for b in tracks:
                for style in ('build','smooth'):
                    score,_=transition_reward(a,b,style)
                    self.assertGreaterEqual(score,0);self.assertLessEqual(score,1)
    def test_planner_dominates_greedy(self):
        tracks=[Track('a',120,'8A',.4),Track('b',122,'8A',.52),Track('c',124,'9A',.63),Track('d',126,'9A',.78)]
        best,route=optimal_route(tracks,horizon=3)
        greedy=greedy_route(tracks,horizon=3)
        self.assertGreaterEqual(best+1e-9,evaluate_route(tracks,greedy)[0])
        self.assertEqual(len(set(route)),len(route))

class TestGymnasium(unittest.TestCase):
    def test_episode_and_mask(self):
        try:from moodmix_env.song_selection.selection_env import SongSelectionEnv
        except ImportError:self.skipTest('Install gymnasium numpy for environment tests')
        tracks=[Track('a',120,'8A',.4),Track('b',122,'8A',.5),Track('c',124,'9A',.6),Track('d',126,'9A',.7)]
        env=SongSelectionEnv(tracks,horizon=3)
        obs,_=env.reset(seed=0)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(obs['action_mask'].tolist(),[0,1,1,1])
        for i,a in enumerate([1,2,3]):
            obs,reward,done,truncated,info=env.step(a)
            self.assertEqual(done,i==2)
            self.assertFalse(truncated)
            self.assertIn('harmonic',info)
        with self.assertRaises(RuntimeError):env.step(0)
        obs,_=env.reset()
        with self.assertRaises(ValueError):env.step(0)

if __name__=='__main__':unittest.main()
