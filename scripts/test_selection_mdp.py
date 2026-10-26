"""Run from repository root: python scripts/test_selection_mdp.py"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moodmix_env.song_selection.rewards import Track,transition_reward
from moodmix_env.song_selection.baselines import greedy_route,optimal_route,evaluate_route

TRACKS=[Track('Warmup House',120,'8A',0.40),Track('Deep Flow',122,'8A',0.52),
        Track('Night Groove',124,'9A',0.63),Track('Peak Drive',126,'9A',0.78),
        Track('Bright Shift',125,'8B',0.70),Track('Dark Pulse',128,'2A',0.92)]

def main():
    source=Track('Current',124,'8A',0.60)
    examples=[Track('Candidate 1',126,'8A',0.70),Track('Candidate 2',128,'9A',0.75),
              Track('Candidate 3',124,'8B',0.60),Track('Candidate 4',140,'2A',0.90)]
    print('REWARD SPECIFICATION EXAMPLES (build):')
    print('Track         Harmonic  BPM     Energy  Total')
    for target in examples:
        r,p=transition_reward(source,target,'build')
        print(f'{target.name:<13} {p["harmonic"]:.3f}     {p["bpm"]:.3f}   {p["energy"]:.3f}   {r:.3f}')
    for style in ('build','smooth'):
        optimal,route=optimal_route(TRACKS,style=style)
        greedy=greedy_route(TRACKS,style=style)
        greedy_total,_=evaluate_route(TRACKS,greedy,style=style)
        print(f'\nSTYLE {style.upper()}')
        for name,path,total in [('Optimal',route,optimal),('Greedy',greedy,greedy_total)]:
            print(f'{name}: '+ ' -> '.join([TRACKS[0].name]+[TRACKS[i].name for i in path])+f' | discounted reward {total:.4f}')
        for row in evaluate_route(TRACKS,route,style=style)[1]:
            print(f'  {row["from"]} -> {row["to"]}: harmonic={row["harmonic"]:.3f}, bpm={row["bpm"]:.3f}, energy={row["energy"]:.3f}, total={row["total_reward"]:.3f}')
    try:
        from moodmix_env.song_selection.selection_env import SongSelectionEnv
    except ImportError:
        print('\nGymnasium not installed: environment smoke test skipped. Install with pip install gymnasium numpy')
    else:
        env=SongSelectionEnv(TRACKS)
        obs,info=env.reset(seed=42)
        assert env.observation_space.contains(obs)
        for action in optimal_route(TRACKS)[1]:
            obs,reward,done,truncated,info=env.step(action)
            assert env.observation_space.contains(obs)
        assert done and not truncated
        print('\nGymnasium reset/step smoke test PASSED')
if __name__=='__main__':main()
