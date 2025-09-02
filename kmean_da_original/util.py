# util.py

import numpy as np

def construct_p_q_t(a, b, A):
    """Constructs the parameters p, q, and o for quadratic inequalities."""
    p = np.dot(a.T, A).item()
    q = np.dot(b.T, A).item()
    o = np.dot(b.T, b).item()
    return p, q, o

def interval_intersection(interval1, interval2):
    """Computes the intersection of two intervals."""
    start1, end1 = interval1
    start2, end2 = interval2
    start = max(start1, start2)
    end = min(end1, end2)
    if start <= end:
        return (start, end)
    else:
        return (-np.inf, np.inf)  # No intersection

def solve_quadratic_inequality(p, q, o):
    """Solves the quadratic inequality of the form p*x^2 + q*x + o <= 0."""
    if p == 0:
        if q == 0:
            return "No solution" if o > 0 else (-np.inf, np.inf)
        else:
            return (-np.inf, -o/q) if q > 0 else (-o/q, np.inf)
    
    discriminant = q**2 - 4*p*o
    if discriminant < 0:
        return "No solution"
    elif discriminant == 0:
        root = -q / (2*p)
        return (root, root)
    else:
        sqrt_disc = np.sqrt(discriminant)
        root1 = (-q - sqrt_disc) / (2*p)
        root2 = (-q + sqrt_disc) / (2*p)
        return (min(root1, root2), max(root1, root2))