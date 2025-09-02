# intersection.py

import numpy as np

def interval_intersection(interval1, interval2):
    """Compute the intersection of two intervals."""
    start1, end1 = interval1
    start2, end2 = interval2
    start = max(start1, start2)
    end = min(end1, end2)
    if start <= end:
        return (start, end)
    else:
        return None

def solve_quadratic_inequality(a, b, c):
    """Solve the quadratic inequality ax^2 + bx + c <= 0."""
    if a == 0:
        if b == 0:
            return None if c > 0 else [(-np.inf, np.inf)]
        else:
            root = -c / b
            return [(-np.inf, root)] if b > 0 else [(root, np.inf)]
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    elif discriminant == 0:
        root = -b / (2*a)
        return [(root, root)] if a > 0 else None
    else:
        sqrt_disc = np.sqrt(discriminant)
        root1 = (-b - sqrt_disc) / (2*a)
        root2 = (-b + sqrt_disc) / (2*a)
        if a > 0:
            return [(root1, root2)]
        else:
            return [(-np.inf, root1), (root2, np.inf)]