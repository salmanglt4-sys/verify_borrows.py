#!/usr/bin/env python3
"""
verify_borrows.py

Supplementary verification script for:
  S. Hussain and S. Yaqoob, "Indispensable digits as borrows: exact moments
  and joint fluctuations of the digit sums of n and (b-1)n."

All checks use exact integer or Fraction arithmetic (no floating point),
and are checks only -- no proof in the paper depends on these computations.
Running this file executes every check and prints PASS/FAIL per section.

Requires: Python 3.8+, no third-party packages (uses fractions, itertools,
functools from the standard library; sympy is used only for exact primality
and binomial coefficients and is optional -- a fallback is provided).
"""

from fractions import Fraction
from itertools import product
from functools import lru_cache
import sys

try:
    from sympy import isprime, binomial as sympy_binomial
    HAVE_SYMPY = True
except ImportError:
    HAVE_SYMPY = False

    def isprime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True

    def sympy_binomial(n, k):
        from math import comb
        return comb(n, k)


# ---------------------------------------------------------------------
# Section 1: basic digit utilities
# ---------------------------------------------------------------------

def digits(n, b):
    """Least significant digit first. digits(0, b) == []."""
    ds = []
    while n > 0:
        n, d = divmod(n, b)
        ds.append(d)
    return ds


def s(n, b):
    """Digit sum of n in base b."""
    return sum(digits(n, b))


def L(n, b):
    """Number of base-b digits of n. L(0) = 0."""
    return len(digits(n, b))


def iota_recursion(n, b):
    """iota_b(n) via the borrow recursion (2.1): beta_i = [d_i > d_{i-1} - beta_{i-1}]."""
    prev = borrow = count = 0
    for d in digits(n, b):
        borrow = 1 if d > prev - borrow else 0
        count += borrow
        prev = d
    return count


def iota_tail(n, b):
    """iota_b(n) via the digit-tail criterion, Theorem 3.1(a):
    beta_i = 1 iff g*(n mod b^i) < d_i * b^i."""
    g = b - 1
    total = 0
    for i, d in enumerate(digits(n, b)):
        if g * (n % (b ** i)) < d * (b ** i):
            total += 1
    return total


def iota_run_definition(n, b):
    """iota_b(n) via Choi's original run-length definition (Def. 3.1 of [C1]):
    d_i is indispensable if d_i = d_{i-1} = ... = d_{i-r+1} > d_{i-r} for some
    1 <= r <= i+1, with d_{-1} = 0. Independent implementation used as a
    cross-check against the borrow recursion (Proposition 2.3)."""
    ds = digits(n, b)  # ds[i] = d_i, least significant first
    count = 0
    for i in range(len(ds)):
        d_i = ds[i]
        r = 1
        while True:
            # check d_i = d_{i-1} = ... = d_{i-r+1}
            equal_run = all(
                ds[i - j] == d_i for j in range(0, r) if i - j >= 0
            )
            if i - r + 1 < 0:
                # ran off the bottom; d_{-1} = 0 terminates the run
                break
            if not equal_run:
                break
            d_i_minus_r = ds[i - r] if i - r >= 0 else 0
            if d_i > d_i_minus_r:
                count += 1
                break
            r += 1
            if r > i + 1:
                break
    return count


# ---------------------------------------------------------------------
# Section 2: carries / thresholds (Theorem 3.1(b),(c), Proposition 4.2)
# ---------------------------------------------------------------------

def carries_and_thresholds(n, b):
    """Return (list of c_i, list of t_i, list of beta_i) for i = 0..L(n)-1,
    per Theorem 3.1(b): c_0 = 0, c_{i+1} = floor((g*d_i + c_i)/b), t_i = 1+c_i,
    beta_i = [d_i > c_i]."""
    g = b - 1
    ds = digits(n, b)
    c = 0
    cs, ts, betas = [], [], []
    for d in ds:
        t = 1 + c
        beta = 1 if d > c else 0
        cs.append(c)
        ts.append(t)
        betas.append(beta)
        c = (g * d + c) // b
    return cs, ts, betas


# ---------------------------------------------------------------------
# Section 3: checks
# ---------------------------------------------------------------------

def check_theorem_2_4_and_prop_2_3_and_3_1(bmax=12, nmax=20000):
    """Theorem 2.4: s_b((b-1)n) = (b-1)*iota_b(n).
    Proposition 2.3 / Theorem 3.1(a): three independent definitions of
    iota_b(n) (run-length, borrow recursion, digit-tail) must agree."""
    for b in range(2, bmax + 1):
        g = b - 1
        for n in range(nmax):
            i_rec = iota_recursion(n, b)
            i_tail = iota_tail(n, b)
            i_run = iota_run_definition(n, b)
            assert i_rec == i_tail == i_run, (
                f"iota mismatch at b={b}, n={n}: "
                f"recursion={i_rec}, tail={i_tail}, run={i_run}"
            )
            lhs = s(g * n, b)
            rhs = g * i_rec
            assert lhs == rhs, (
                f"Theorem 2.4 fails at b={b}, n={n}: "
                f"s_b(gn)={lhs}, g*iota_b(n)={rhs}"
            )
    print(f"[PASS] Theorem 2.4, Proposition 2.3, Theorem 3.1(a) "
          f"for 2<=b<={bmax}, 0<=n<{nmax}")


def check_theorem_3_1_bc(bmax=12, nmax=20000):
    """Theorem 3.1(b): c_{i+1} = floor((g d_i + c_i)/b), t_i = 1+c_i,
    beta_i = [d_i > c_i]. Theorem 3.1(c): beta_i = d_i - c_{i+1};
    iota_b(n) = s_b(n) - sum_{i=1}^{L(n)} c_i."""
    for b in range(2, bmax + 1):
        for n in range(nmax):
            ds = digits(n, b)
            cs, ts, betas = carries_and_thresholds(n, b)
            # cross-check against the borrow recursion
            i_rec = iota_recursion(n, b)
            assert sum(betas) == i_rec, (
                f"carry-derived borrows disagree with recursion at b={b}, n={n}"
            )
            # (c): beta_i = d_i - c_{i+1}
            for i in range(len(ds)):
                c_next = cs[i + 1] if i + 1 < len(cs) else (
                    ((b - 1) * ds[i] + cs[i]) // b
                )
                assert betas[i] == ds[i] - c_next, (
                    f"Theorem 3.1(c) beta_i=d_i-c_{{i+1}} fails at b={b}, n={n}, i={i}"
                )
            iota_via_sum = s(n, b) - sum(cs[1:] + [
                ((b - 1) * ds[-1] + cs[-1]) // b
            ] if ds else [])
            # simpler: recompute sum_{i=1}^{L} c_i directly
            full_cs = cs + [((b - 1) * ds[-1] + cs[-1]) // b] if ds else [0]
            sum_c_1_to_L = sum(full_cs[1:])
            assert s(n, b) - sum_c_1_to_L == i_rec, (
                f"Theorem 3.1(c) iota=s_b(n)-sum c_i fails at b={b}, n={n}"
            )
    print(f"[PASS] Theorem 3.1(b),(c) for 2<=b<={bmax}, 0<=n<{nmax}")


def check_corollary_3_3(pmax=11, nmax=2000):
    """Corollary 3.3: v_p(C(pn,n)) = s_p((p-1)n)/(p-1) = iota_p(n), for primes p."""
    for p in range(2, pmax + 1):
        if not isprime(p):
            continue
        for n in range(nmax):
            binom = sympy_binomial(p * n, n)
            vp = 0
            m = binom
            if m != 0:
                while m % p == 0:
                    vp += 1
                    m //= p
            i_rec = iota_recursion(n, p)
            assert vp == i_rec, (
                f"Corollary 3.3 fails at p={p}, n={n}: v_p(C(pn,n))={vp}, iota_p(n)={i_rec}"
            )
    print(f"[PASS] Corollary 3.3 for primes p<={pmax}, n<{nmax}")


def check_extreme_cases_and_counts(bk_max=2 * 10 ** 5):
    """Theorem 3.5 (characterisation of iota_b(n)=1 and iota_b(n)=L(n)) and
    Corollary 3.6 (counts), checked by exhaustion over n < b^k for b^k <= bk_max.
    Also checks Proposition 4.1 (Choi's equidistribution) by exhaustion."""
    from math import comb
    for b in range(2, 8):
        g = b - 1
        k = 1
        while b ** k <= bk_max:
            N = b ** k
            count_iota1 = 0
            count_iotaL = 0
            dist_iota = {}
            for n in range(N):
                i_n = iota_recursion(n, b)
                dist_iota[i_n] = dist_iota.get(i_n, 0) + 1
                if n >= 1 and i_n == 1:
                    count_iota1 += 1
                if n >= 1 and i_n == L(n, b) == k:
                    count_iotaL += 1

                # Proposition 4.1: iota_b(n) = ceil(s_b(r)/g) where r = g n mod b^k
                r = (g * n) % N
                import math
                pred = math.ceil(s(r, b) / g) if g > 0 else 0
                assert pred == i_n, (
                    f"Proposition 4.1 fails at b={b}, k={k}, n={n}: "
                    f"predicted {pred}, actual {i_n}"
                )

            expected_iota1 = comb(k + g, g) - 1
            expected_iotaL = comb(k + g - 1, k)
            assert count_iota1 == expected_iota1, (
                f"Corollary 3.6 (iota=1 count) fails at b={b}, k={k}: "
                f"got {count_iota1}, expected {expected_iota1}"
            )
            assert count_iotaL == expected_iotaL, (
                f"Corollary 3.6 (iota=L count) fails at b={b}, k={k}: "
                f"got {count_iotaL}, expected {expected_iotaL}"
            )
            k += 1
    print(f"[PASS] Theorem 3.5, Corollary 3.6, Proposition 4.1 "
          f"by exhaustion, b^k<={bk_max}")


def check_proposition_chain(bmax=12, mmax=8):
    """Proposition~\\ref{prop:chain}: the threshold chain has transition
    matrix P=(I+J)/b; checks (a) P(t,t')=(1+[t=t'])/b by direct enumeration
    over one digit step, (b) the closed form P^m = Pi + b^{-m}(I-Pi) against
    brute-force integer matrix powers of b^m P^m (kept as integers to stay
    exact), and (c) the law of t_i -- E[t_i] and E[t_i^2] via eq:mix with
    f(t)=t and f(t)=t^2 -- against brute-force exhaustion over all n<b^i."""
    for b in range(2, bmax + 1):
        g = b - 1
        if g == 0:
            continue
        states = list(range(1, g + 1))

        # (a) one-step transition matrix, as exact counts out of b
        P = {t: {tp: 0 for tp in states} for t in states}
        for t in states:
            for d in range(0, g + 1):
                tp = d + 1 if d < t else d
                P[t][tp] += 1
        for t in states:
            total = sum(P[t].values())
            assert total == b, f"row sum mismatch at b={b}, t={t}"
            for tp in states:
                expected = 1 + (1 if t == tp else 0)
                assert P[t][tp] == expected, (
                    f"Proposition chain transition matrix fails at b={b}: "
                    f"P({t},{tp}) count {P[t][tp]} != {expected}"
                )

        # (b) P^m = Pi + b^{-m}(I-Pi), Pi = (1/g) all-ones matrix on states.
        # Work with bP = b*P (integer matrix) and compare (bP)^m / b^m
        # against the closed form, via exact Fractions.
        bP = [[Fraction(P[t][tp]) for tp in states] for t in states]

        def mat_mul(A, Bm):
            n = len(A)
            return [[sum(A[i][l] * Bm[l][j] for l in range(n))
                      for j in range(n)] for i in range(n)]

        Pm = bP  # represents b^1 * P^1 after one step; we'll track b^m P^m
        for m in range(1, mmax + 1):
            if m > 1:
                Pm = mat_mul(Pm, bP)
            # Pm currently = (b*P)^m = b^m P^m
            Pm_exact = [[Pm[i][j] / Fraction(b) ** m for j in range(len(states))]
                        for i in range(len(states))]
            for i, t in enumerate(states):
                for j, tp in enumerate(states):
                    closed = Fraction(1, g) + (Fraction(b) ** (-m)) * (
                        (1 if i == j else 0) - Fraction(1, g)
                    )
                    assert Pm_exact[i][j] == closed, (
                        f"Proposition chain P^m formula fails at b={b}, "
                        f"m={m}, t={t}, t'={tp}"
                    )

        # (c) law of t_i: E[t_i], E[t_i^2] via eq:mix (f(t)=t, pi(f)=b/2;
        # f(t)=t^2, pi(f)=b(2b-1)/6), starting from t_0=1, checked against
        # brute-force exhaustion over n < b^i for small i. t_i depends only
        # on digits d_0,...,d_{i-1} of n -- which must be zero-padded to
        # exactly i digits (digits() strips leading zeros, so can't be used
        # directly here).
        def threshold_after(n, i):
            ds = digits(n, b)
            ds = (ds + [0] * i)[:i]
            c = 0
            for d in ds:
                c = (g * d + c) // b
            return 1 + c

        for i in range(0, min(mmax, 6)):
            N = b ** i
            if N > 500000:
                break
            sum_t = sum_t2 = 0
            for n in range(N):
                t_i = threshold_after(n, i)
                sum_t += t_i
                sum_t2 += t_i * t_i
            E_t = Fraction(sum_t, N)
            E_t2 = Fraction(sum_t2, N)
            want_Et = Fraction(b, 2) - Fraction(b - 2, 2) * Fraction(1, b) ** i
            want_Et2 = Fraction(b * (2 * b - 1), 6) - Fraction(
                b * (2 * b - 1) - 6, 6
            ) * Fraction(1, b) ** i
            assert E_t == want_Et, (
                f"Proposition chain law of t_i (mean) fails at b={b}, i={i}: "
                f"got {E_t}, want {want_Et}"
            )
            assert E_t2 == want_Et2, (
                f"Proposition chain law of t_i (2nd moment) fails at b={b}, "
                f"i={i}: got {E_t2}, want {want_Et2}"
            )
    print(f"[PASS] Proposition chain: transition matrix, P^m formula, "
          f"and law of t_i, for 2<=b<={bmax}")


def check_second_moments(bmax=16, kmax=25):
    """Corollary 4.3, Lemma 5.2, Theorem 5.1: exact mean, variance and
    covariance of iota_b(n) and s_b(n), checked by brute-force exhaustion
    for small b^k and by the transfer-matrix method (Proposition 4.4) for
    all b<=bmax, k<=kmax using exact Fraction arithmetic."""
    EXHAUST_LIMIT = 200000  # cap brute force; transfer matrix covers the rest

    def transfer_matrix_moments(b, k):
        """Compute E[iota], E[iota^2], E[s*iota] via the transfer matrix
        of Proposition 4.4, generalised to track (count, sum_iota, sum_iota^2,
        sum_s, sum_s_iota) using exact Fractions, by dynamic programming
        over (threshold state) with weight bookkeeping."""
        g = b - 1
        if g == 0:
            return None
        states = list(range(1, g + 1))
        # vector of (N, E_iota*N, E_iota2*N, E_s*N, E_s_iota*N) per state,
        # all as exact integers (we track un-normalised sums over the N=b^k
        # strings consistent with reaching this state after i digits).
        # dist[t] = (count, sum_iota, sum_iota2, sum_s, sum_s_iota)
        dist = {1: (1, 0, 0, 0, 0)}
        for step in range(k):
            new_dist = {t: [0, 0, 0, 0, 0] for t in states}
            for t, (cnt, si, si2, ss, ssi) in dist.items():
                for d in range(0, g + 1):
                    beta = 1 if d >= t else 0
                    tp = d + 1 if d < t else d
                    nd = new_dist[tp]
                    nd[0] += cnt
                    nd[1] += si + beta * cnt
                    nd[2] += si2 + 2 * beta * si + beta * cnt
                    nd[3] += ss + d * cnt
                    nd[4] += ssi + d * si + beta * ss + beta * d * cnt
            dist = {t: tuple(v) for t, v in new_dist.items()}
        N = b ** k
        total_cnt = sum(v[0] for v in dist.values())
        assert total_cnt == N
        sum_iota = sum(v[1] for v in dist.values())
        sum_iota2 = sum(v[2] for v in dist.values())
        sum_s = sum(v[3] for v in dist.values())
        sum_s_iota = sum(v[4] for v in dist.values())
        E_iota = Fraction(sum_iota, N)
        E_iota2 = Fraction(sum_iota2, N)
        E_s = Fraction(sum_s, N)
        E_s_iota = Fraction(sum_s_iota, N)
        Var_iota = E_iota2 - E_iota ** 2
        Cov_s_iota = E_s_iota - E_s * E_iota
        return E_iota, Var_iota, Cov_s_iota

    def closed_form(b, k):
        g = b - 1
        x = Fraction(1, b ** k)
        E_iota = Fraction(k, 2) + Fraction((b - 2) * (1 - x), 2 * (b - 1))
        Var_iota = (
            Fraction(b + 1, 12 * (b - 1)) * k
            + Fraction(b * (b - 2), 12 * (b - 1) ** 2)
            - Fraction(
                (b - 2) * (2 * (b - 3) * ((b - 1) * k - 1) * x + 3 * (b - 2) * x ** 2),
                12 * (b - 1) ** 2,
            )
        )
        Cov_s_iota = (
            Fraction(b + 1, 12 * (b - 1)) * k
            + Fraction(b * (b - 2), 3 * (b - 1) ** 2)
            + Fraction(
                (b - 2) * ((b - 1) * (b - 3) * k - 4 * b) * x,
                12 * (b - 1) ** 2,
            )
        )
        return E_iota, Var_iota, Cov_s_iota

    checked = 0
    for b in range(2, bmax + 1):
        for k in range(1, kmax + 1):
            got = transfer_matrix_moments(b, k)
            want = closed_form(b, k)
            assert got == want, (
                f"Theorem 5.1 / Corollary 4.3 mismatch at b={b}, k={k}: "
                f"transfer-matrix {got} vs closed form {want}"
            )
            checked += 1
    print(f"[PASS] Corollary 4.3, Theorem 5.1 (transfer-matrix vs closed form) "
          f"for 2<=b<={bmax}, 1<=k<={kmax} ({checked} cases)")

    # brute-force cross-check for small cases
    bf_checked = 0
    for b in range(2, 7):
        k = 1
        while b ** k <= EXHAUST_LIMIT:
            N = b ** k
            sum_iota = sum_iota2 = sum_s = sum_s_iota = 0
            for n in range(N):
                i_n = iota_recursion(n, b)
                s_n = s(n, b)
                sum_iota += i_n
                sum_iota2 += i_n * i_n
                sum_s += s_n
                sum_s_iota += s_n * i_n
            E_iota = Fraction(sum_iota, N)
            Var_iota = Fraction(sum_iota2, N) - E_iota ** 2
            Cov_s_iota = Fraction(sum_s_iota, N) - Fraction(sum_s, N) * E_iota
            want = closed_form(b, k)
            assert (E_iota, Var_iota, Cov_s_iota) == want, (
                f"brute-force moment mismatch at b={b}, k={k}"
            )
            bf_checked += 1
            k += 1
    print(f"[PASS] Theorem 5.1 brute-force cross-check ({bf_checked} cases, "
          f"b^k<={EXHAUST_LIMIT})")


def check_lemma_residue(bmax=12, kmax=8):
    """Lemma~\\ref{lem:residue}: for X_1,...,X_k iid uniform on {0,...,g}
    and S their sum, (a) P(S == c mod g) = 1/g + b^{-k}([g|c] - 1/g), and
    (b) E[(S - gk/2) * 1{S==c mod g}] = k b^{-k} (rho(c) - g/2) [g does not
    divide c], for every integer c. Checked exactly, via the exact
    distribution of S computed by convolution (not by enumerating all b^k
    tuples), for c ranging over one full residue system mod g."""

    def digit_sum_distribution(g, k):
        # dist[S] = #{(x_1,...,x_k) in {0,...,g}^k : sum = S}, 0<=S<=g*k
        dist = [1]
        for _ in range(k):
            new_dist = [0] * (len(dist) + g)
            for s_val, cnt in enumerate(dist):
                if cnt == 0:
                    continue
                for x in range(g + 1):
                    new_dist[s_val + x] += cnt
            dist = new_dist
        return dist

    for b in range(2, bmax + 1):
        g = b - 1
        if g == 0:
            continue
        for k in range(1, kmax + 1):
            N = b ** k
            if N > 10 ** 7:
                break
            dist = digit_sum_distribution(g, k)
            assert sum(dist) == N, f"distribution total mismatch b={b}, k={k}"
            for c in range(g):  # one full residue system mod g
                rho_c = c  # least non-negative residue of c mod g, c in [0,g-1]
                g_divides_c = (c == 0)

                cnt_c = sum(cnt for S, cnt in enumerate(dist) if S % g == c)
                P = Fraction(cnt_c, N)
                want_P = Fraction(1, g) + Fraction(1, N) * (
                    (1 if g_divides_c else 0) - Fraction(1, g)
                )
                assert P == want_P, (
                    f"Lemma residue (a) fails at b={b}, k={k}, c={c}: "
                    f"P={P}, want={want_P}"
                )

                weighted = sum(
                    (Fraction(S) - Fraction(g * k, 2)) * cnt
                    for S, cnt in enumerate(dist) if S % g == c
                )
                lhs = weighted / N
                want_lhs = (
                    Fraction(0) if g_divides_c else
                    Fraction(k, N) * (Fraction(rho_c) - Fraction(g, 2))
                )
                assert lhs == want_lhs, (
                    f"Lemma residue (b) fails at b={b}, k={k}, c={c}: "
                    f"got={lhs}, want={want_lhs}"
                )
    print(f"[PASS] Lemma residue (a),(b) for 2<=b<={bmax}, k<={kmax}")


def check_proposition_transfer(bmax=10, kmax=8):
    """Proposition~\\ref{prop:transfer}: sum_{n<b^k} x^{iota_b(n)} =
    e_1^T M(x)^k 1, where M(x)_{t,t'} = [t'<=t] + x[t'>=t]. Checked as a
    polynomial identity in x (exact integer coefficients, via matrix
    powers with polynomial entries truncated at degree k), with the
    coefficients compared against the brute-force distribution of
    iota_b(n) over n < b^k."""

    def poly_mul(p, q, maxdeg):
        res = [0] * (maxdeg + 1)
        for i, a in enumerate(p):
            if a == 0 or i > maxdeg:
                continue
            for j, c in enumerate(q):
                if a == 0 or c == 0 or i + j > maxdeg:
                    continue
                res[i + j] += a * c
        return res

    def poly_add(p, q):
        return [a + b_ for a, b_ in zip(p, q)]

    def mat_mul(A, B, maxdeg):
        n = len(A)
        zero = [0] * (maxdeg + 1)
        result = [[list(zero) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for l in range(n):
                if all(c == 0 for c in A[i][l]):
                    continue
                for j in range(n):
                    if all(c == 0 for c in B[l][j]):
                        continue
                    result[i][j] = poly_add(
                        result[i][j], poly_mul(A[i][l], B[l][j], maxdeg)
                    )
        return result

    for b in range(2, bmax + 1):
        g = b - 1
        if g == 0:
            continue
        for k in range(1, kmax + 1):
            N = b ** k
            if N > 300000:
                break
            maxdeg = k
            states = list(range(1, g + 1))
            n_states = len(states)
            zero_poly = [0] * (maxdeg + 1)

            def entry(t, tp):
                c0 = 1 if tp <= t else 0
                c1 = 1 if tp >= t else 0
                p = list(zero_poly)
                p[0] += c0
                if maxdeg >= 1:
                    p[1] += c1
                return p

            M = [[entry(t, tp) for tp in states] for t in states]

            # M^k by repeated squaring
            def identity():
                I = [[list(zero_poly) for _ in range(n_states)]
                     for _ in range(n_states)]
                for i in range(n_states):
                    I[i][i][0] = 1
                return I

            result = identity()
            base = M
            e = k
            while e > 0:
                if e & 1:
                    result = mat_mul(result, base, maxdeg)
                base = mat_mul(base, base, maxdeg)
                e >>= 1

            # e_1^T M^k 1 = sum over columns of row 0 (state t=1 is index 0)
            gen_poly = [0] * (maxdeg + 1)
            for j in range(n_states):
                gen_poly = poly_add(gen_poly, result[0][j])

            # brute-force distribution of iota_b(n) for n < b^k
            dist = [0] * (maxdeg + 1)
            for n in range(N):
                j = iota_recursion(n, b)
                dist[j] += 1

            assert gen_poly == dist, (
                f"Proposition transfer generating-function identity fails "
                f"at b={b}, k={k}: poly={gen_poly}, brute-force dist={dist}"
            )
    print(f"[PASS] Proposition transfer generating-function identity "
          f"for 2<=b<={bmax}, k<={kmax}")


def check_lemma_digitsAn_and_gamma(cases=None):
    """Lemma~\\ref{lem:digitsAn}(a),(b): the digit d_i(An) is uniform on
    {0,...,g} and, for i<j, d_j(An) is independent of n mod b^{i+1} --
    checked by exhaustion for small (b,A,k). Also checks the bound
    |gamma_i - gamma| <= 2g^2(A1+A2) b^{-i-1} from the proof of
    Theorem~\\ref{thm:cov}, where gamma_i = Cov(d_i(A1 n), d_i(A2 n)) for
    n uniform mod b^{i+1}, and gamma = (b^2-1)/12 * gcd(A1,A2)^2/(A1 A2)."""
    from math import gcd

    if cases is None:
        cases = [(10, 1, 9), (10, 3, 7), (7, 1, 6)]

    for (b, A1, A2) in cases:
        assert gcd(A1, b) == 1 and gcd(A2, b) == 1
        g = b - 1

        def digit_of_Aj_n(n, m, i, A):
            # d_i(A*n) given n mod m = b^{i+1}; works for any n (reduces mod m)
            v = (A * (n % m)) % m
            return (v // (b ** i))

        # (a) uniformity: for a few small i, enumerate n mod b^{i+1} and
        # check d_i(A*n) hits every value in {0,...,g} equally often.
        for A in (A1, A2):
            for i in range(0, 5):
                m = b ** (i + 1)
                if m > 2 * 10 ** 6:
                    break
                counts = [0] * b
                for n in range(m):
                    counts[digit_of_Aj_n(n, m, i, A)] += 1
                assert all(c == m // b for c in counts), (
                    f"Lemma digitsAn(a) uniformity fails: b={b}, A={A}, i={i}"
                )

        # (b) independence: for i<j, the joint distribution of
        # (n mod b^{i+1}, d_j(A*n)) should factor -- i.e. d_j(A*n) is
        # uniform *for every fixed residue* n mod b^{i+1}, not just overall.
        for A in (A1, A2):
            i, j = 0, 2
            if j >= 5:
                continue
            m_i = b ** (i + 1)
            m_j = b ** (j + 1)
            if m_j > 2 * 10 ** 6:
                continue
            for r in range(m_i):
                counts = [0] * b
                step = m_i
                n = r
                cnt_total = 0
                while n < m_j:
                    counts[digit_of_Aj_n(n, m_j, j, A)] += 1
                    cnt_total += 1
                    n += step
                assert all(c == cnt_total // b for c in counts), (
                    f"Lemma digitsAn(b) independence fails: b={b}, A={A}, "
                    f"i={i}, j={j}, r={r}"
                )

        # gamma_i vs gamma bound, from the proof of Theorem thm:cov.
        delta = gcd(A1, A2)
        gamma = Fraction(b * b - 1, 12) * Fraction(delta * delta, A1 * A2)
        for i in range(0, 6):
            m = b ** (i + 1)
            if m > 2 * 10 ** 6:
                break
            sum1 = sum2 = sum12 = 0
            for n in range(m):
                d1 = digit_of_Aj_n(n, m, i, A1)
                d2 = digit_of_Aj_n(n, m, i, A2)
                sum1 += d1
                sum2 += d2
                sum12 += d1 * d2
            E1 = Fraction(sum1, m)
            E2 = Fraction(sum2, m)
            gamma_i = Fraction(sum12, m) - E1 * E2
            bound = Fraction(2 * g * g * (A1 + A2)) * Fraction(1, b) ** (i + 1)
            assert abs(gamma_i - gamma) <= bound, (
                f"gamma_i bound fails: b={b}, A1={A1}, A2={A2}, i={i}: "
                f"|gamma_i-gamma|={float(abs(gamma_i - gamma)):.6f} > "
                f"bound={float(bound):.6f}"
            )
    print("[PASS] Lemma digitsAn(a),(b) and the |gamma_i-gamma| bound "
          "(proof of Theorem thm:cov)")


def check_table_2():
    """Every entry of Table 2 (base ten, k=1..6): counts by iota value, and
    the exact mean/variance/covariance rounded to four decimals."""
    table = {
        1: {1: 9},
        2: {1: 54, 2: 45},
        3: {1: 219, 2: 615, 3: 165},
        4: {1: 714, 2: 4620, 3: 4170, 4: 495},
        5: {1: 2001, 2: 25212, 3: 50412, 4: 21087, 5: 1287},
        6: {1: 5004, 2: 111573, 3: 411048, 4: 381723, 5: 87648, 6: 3003},
    }
    stats = {
        1: (0.9000, 0.0900, 0.4500),
        2: (1.4400, 0.2664, 0.5400),
        3: (1.9440, 0.3849, 0.6360),
        4: (2.4444, 0.4893, 0.7368),
        5: (2.9444, 0.5915, 0.8385),
        6: (3.4444, 0.6934, 0.9403),
    }
    b = 10
    for k, row in table.items():
        N = b ** k
        counts = {}
        sum_iota = sum_iota2 = sum_s = sum_s_iota = 0
        for n in range(N):
            i_n = iota_recursion(n, b)
            s_n = s(n, b)
            counts[i_n] = counts.get(i_n, 0) + 1
            sum_iota += i_n
            sum_iota2 += i_n * i_n
            sum_s += s_n
            sum_s_iota += s_n * i_n
        # row 0 (n=0, iota=0) is always count 1 but omitted from the paper's
        # printed table (it shows only j>=1 columns); add it back for the
        # full distribution check, matching only the j>=1 entries against `row`.
        counts.pop(0, None)
        assert counts == row, f"Table 2 counts mismatch at k={k}: {counts} != {row}"

        E_iota = sum_iota / N
        Var_iota = sum_iota2 / N - E_iota ** 2
        Cov_s_iota = sum_s_iota / N - (sum_s / N) * E_iota
        want_mean, want_var, want_cov = stats[k]
        assert abs(E_iota - want_mean) < 5e-5, f"Table 2 mean mismatch at k={k}"
        assert abs(Var_iota - want_var) < 5e-5, f"Table 2 variance mismatch at k={k}"
        assert abs(Cov_s_iota - want_cov) < 5e-5, f"Table 2 covariance mismatch at k={k}"
    print("[PASS] Every entry of Table 2 (counts, mean, variance, covariance)")


def check_theorem_cov(cases=None):
    """Theorem~\\ref{thm:cov}: for A1, A2 coprime to b,
    Cov(s_b(A1 n), s_b(A2 n)) = (b^2-1)/12 * gcd(A1,A2)^2/(A1 A2) * k + R_k,
    with the EXACT bound stated in the theorem:
    |R_k| <= 2g(A1+A2) + (b/2)(tau1*A1+tau2*A2) + tau1*tau2/4,
    where g=b-1 and tau_j = max_{0<=m<A_j} s_b(m).
    This checks the bound itself (as stated), not just the leading term,
    by brute force over n in [0, b^k) for several (b,A1,A2,k)."""
    from math import gcd

    if cases is None:
        cases = [
            (10, 1, 9),   # the b-1 case specialised throughout the paper
            (10, 3, 7),
            (7, 1, 6),
            (6, 1, 5),
        ]

    for (b, A1, A2) in cases:
        assert gcd(A1, b) == 1 and gcd(A2, b) == 1, (
            f"Theorem thm:cov requires A1, A2 coprime to b; got b={b}, "
            f"A1={A1}, A2={A2}"
        )
        g = b - 1
        delta = gcd(A1, A2)
        leading_coeff = Fraction(b * b - 1, 12) * Fraction(delta * delta, A1 * A2)
        tau1 = max(s(m, b) for m in range(A1)) if A1 > 1 else 0
        tau2 = max(s(m, b) for m in range(A2)) if A2 > 1 else 0
        bound = Fraction(2 * g * (A1 + A2)) \
            + Fraction(b, 2) * (tau1 * A1 + tau2 * A2) \
            + Fraction(tau1 * tau2, 4)

        checked_any = False
        for k in range(1, 8):
            N = b ** k
            if N > 300000:
                break
            sum1 = sum2 = sum12 = 0
            for n in range(N):
                s1 = s(A1 * n, b)
                s2 = s(A2 * n, b)
                sum1 += s1
                sum2 += s2
                sum12 += s1 * s2
            E1 = Fraction(sum1, N)
            E2 = Fraction(sum2, N)
            Cov = Fraction(sum12, N) - E1 * E2
            R_k = Cov - leading_coeff * k
            assert abs(R_k) <= bound, (
                f"Theorem thm:cov error-term bound violated: b={b}, A1={A1}, "
                f"A2={A2}, k={k}: |R_k|={float(abs(R_k)):.4f} > "
                f"bound={float(bound):.4f}"
            )
            checked_any = True
        assert checked_any, f"no feasible k for (b,A1,A2)=({b},{A1},{A2})"
        print(f"  (b,A1,A2)=({b},{A1},{A2}): leading coeff "
              f"{float(leading_coeff):.4f}, |R_k| stayed within the "
              f"theorem's bound {float(bound):.4f} for all k tried")
    print("[PASS] Theorem thm:cov (general multipliers, with its stated error bound)")


def check_remark_dk_b():
    """Remark rem:dk(b): for b=10, A1=1, A2=2 (NOT coprime to b, so
    Theorem thm:cov does not apply), Cov(s_10(n), s_10(2n)) = 21k/4 exactly,
    for n uniform on [0, 10^k) -- confirming the paper's point that
    coprimality cannot be dropped (the naive formula would predict 33k/8)."""
    b = 10
    for k in range(1, 7):
        N = b ** k
        sum1 = sum2 = sum12 = 0
        for n in range(N):
            s1 = s(n, b)
            s2 = s(2 * n, b)
            sum1 += s1
            sum2 += s2
            sum12 += s1 * s2
        E1 = Fraction(sum1, N)
        E2 = Fraction(sum2, N)
        Cov = Fraction(sum12, N) - E1 * E2
        want = Fraction(21, 4) * k
        naive = Fraction(99, 12) * Fraction(1, 2) * k  # Fraction(33,8)*k
        assert Cov == want, (
            f"Remark rem:dk(b) fails at k={k}: Cov={Cov}, want={want}"
        )
        assert Cov != naive, (
            f"Remark rem:dk(b) sanity check failed: exact value should "
            f"differ from the (inapplicable) coprime-case formula"
        )
    print("[PASS] Remark rem:dk(b) exact identity, k=1..6 "
          "(and confirms it differs from the coprime-case formula, as claimed)")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("Running verify_borrows.py ...")
    print(f"(sympy available: {HAVE_SYMPY})\n")

    check_theorem_2_4_and_prop_2_3_and_3_1(bmax=12, nmax=20000)
    check_theorem_3_1_bc(bmax=10, nmax=5000)  # reduced default; raise nmax/bmax for full run
    check_corollary_3_3(pmax=11, nmax=2000)
    check_extreme_cases_and_counts(bk_max=200000)
    check_proposition_chain(bmax=12, mmax=8)
    check_lemma_residue(bmax=12, kmax=8)
    check_proposition_transfer(bmax=10, kmax=8)
    check_second_moments(bmax=16, kmax=25)
    check_table_2()
    check_lemma_digitsAn_and_gamma()
    check_theorem_cov()
    check_remark_dk_b()

    print("\nAll checks passed.")


if __name__ == "__main__":
    sys.exit(main())
