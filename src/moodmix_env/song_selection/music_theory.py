"""Camelot-wheel harmonic relationships; no double-counting of fifths."""
import re

def parse_camelot(key: str) -> tuple[int, str]:
    if not isinstance(key, str) or not re.fullmatch(r'(?:[1-9]|1[0-2])[AaBb]', key.strip()):
        raise ValueError(f'Invalid Camelot key: {key!r}; expected 1A..12A or 1B..12B')
    s = key.strip().upper()
    return int(s[:-1]), s[-1]

def circle_distance(a: str, b: str) -> int:
    x, _ = parse_camelot(a)
    y, _ = parse_camelot(b)
    return min(abs(x-y), 12-abs(x-y))

def harmonic_details(a: str, b: str) -> dict:
    na, ma = parse_camelot(a)
    nb, mb = parse_camelot(b)
    d = min(abs(na-nb), 12-abs(na-nb))
    if d == 0 and ma == mb:
        return {'score': 1.00, 'relationship': 'same_key', 'circle_distance': 0}
    if d == 0 and ma != mb:
        return {'score': 0.95, 'relationship': 'relative_major_minor', 'circle_distance': 0}
    if ma == mb and d == 1:
        return {'score': 0.85, 'relationship': 'adjacent_fifth', 'circle_distance': 1}
    if ma == mb and d == 2:
        return {'score': 0.50, 'relationship': 'two_fifths', 'circle_distance': 2}
    return {'score': 0.10, 'relationship': 'other', 'circle_distance': d}

def harmonic_score(a: str, b: str) -> float:
    return harmonic_details(a, b)['score']
