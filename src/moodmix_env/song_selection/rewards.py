"""Version 2 reward specification. Chord reward intentionally not implemented."""
from dataclasses import dataclass
from .music_theory import harmonic_details

@dataclass(frozen=True)
class Track:
    name: str
    bpm: float
    camelot: str
    energy: float

    def __post_init__(self):
        from .music_theory import parse_camelot
        if not self.name.strip():
            raise ValueError('Track name must not be empty')
        if not (0 < self.bpm < 400):
            raise ValueError('BPM must be between 0 and 400')
        parse_camelot(self.camelot)
        if not (0 <= self.energy <= 1):
            raise ValueError('Energy must be between 0 and 1')

WEIGHTS = {'harmonic': 0.45, 'bpm': 0.35, 'energy': 0.20}

def bpm_score(a: float, b: float, tolerance: float = 12.0) -> float:
    if tolerance <= 0:
        raise ValueError('BPM tolerance must be positive')
    return max(0.0, 1.0 - abs(a-b)/tolerance)

def energy_score(a: float, b: float, style: str) -> float:
    if style == 'smooth':
        error = abs(b-a)
    elif style == 'build':
        error = abs((b-a)-0.10)
    else:
        raise ValueError('style must be smooth or build')
    return max(0.0, 1.0-error/0.30)

def transition_reward(a: Track, b: Track, style: str, weights=None, bpm_tolerance=12.0) -> tuple[float, dict]:
    weights = WEIGHTS if weights is None else weights
    if set(weights) != set(WEIGHTS) or any(v < 0 for v in weights.values()) or abs(sum(weights.values())-1) > 1e-9:
        raise ValueError('weights must be nonnegative harmonic/bpm/energy values summing to 1')
    harm = harmonic_details(a.camelot, b.camelot)
    parts = {'harmonic': harm['score'], 'bpm': bpm_score(a.bpm,b.bpm,bpm_tolerance), 'energy': energy_score(a.energy,b.energy,style)}
    total = sum(parts[k]*weights[k] for k in WEIGHTS)
    return total, {**parts, 'relationship':harm['relationship'], 'circle_distance':harm['circle_distance'], 'total_reward':total}
