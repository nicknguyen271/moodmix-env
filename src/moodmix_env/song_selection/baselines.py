"""Comparable greedy and exact finite-horizon planners (small libraries)."""
from functools import lru_cache
from .rewards import transition_reward

def greedy_route(tracks,start=0,horizon=3,style='build'):
    current=start; visited={start}; route=[]
    for _ in range(min(horizon,len(tracks)-1)):
        nxt=max((i for i in range(len(tracks)) if i not in visited), key=lambda i:transition_reward(tracks[current],tracks[i],style)[0])
        route.append(nxt);visited.add(nxt);current=nxt
    return tuple(route)

def optimal_route(tracks,start=0,horizon=3,style='build',gamma=0.95):
    if not 0<gamma<=1:raise ValueError('gamma must be in (0,1]')
    @lru_cache(None)
    def solve(current, visited, left):
        if left==0 or len(visited)==len(tracks):return 0.0,()
        choices=[]
        for nxt in range(len(tracks)):
            if nxt in visited:continue
            r,_=transition_reward(tracks[current],tracks[nxt],style)
            tail,path=solve(nxt,tuple(sorted((*visited,nxt))),left-1)
            choices.append((r+gamma*tail,(nxt,*path)))
        return max(choices,key=lambda pair:pair[0])
    return solve(start,(start,),min(horizon,len(tracks)-1))

def evaluate_route(tracks,route,start=0,style='build',gamma=0.95):
    current=start; total=0.0; rows=[];visited={start}
    for t,nxt in enumerate(route):
        if nxt in visited:raise ValueError('Track replay in route')
        reward,parts=transition_reward(tracks[current],tracks[nxt],style)
        rows.append({'step':t+1,'from':tracks[current].name,'to':tracks[nxt].name,**parts})
        total+=gamma**t*reward
        visited.add(nxt);current=nxt
    return total,rows
