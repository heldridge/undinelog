# Undoing a Stitch with Helpers at Threshold 2κ

The paper's Section 4.3 obtains the missing half of the source state from
maintainers that advertise κ. This document gives a procedure served by
maintainers that advertise 2κ, for use when no registry at threshold κ is
available. It uses the notation of the paper's Section 3: a stitched
segment `P_T` on the parent axes `G^Z_T = G^Z_0 ⊔ G^Z_1` and
`G^W_T = G^W_0 ⊔ G^W_1`, its four sources `P_{μν}`, the selectors `E^Z_μ`
and `E^W_ν` (equation (2)), the stitch `P_T = Σ_{μ,ν} E^Z_μ E^W_ν P_{μν}`
(equation (3)), and the array formulas (4) and (5) of Theorem 1. The
requester has assignments `(C, w)` and holds `L_{T,C}(W) = P_T(C, W)` on
`G^W_T` and `R_{T,w}(Z) = P_T(Z, w)` on `G^Z_T`. Helper `h` has
assignments `(C_h, w_h)` and holds `L_{T,C_h}` and `R_{T,w_h}` likewise.

## Partial stitches

Define

    P_{μ*}(Z, W) = Σ_{ν ∈ {0,1}} E^W_ν(W) P_{μν}(Z, W),
    P_{*ν}(Z, W) = Σ_{μ ∈ {0,1}} E^Z_μ(Z) P_{μν}(Z, W).

`P_{μ*}` has degree less than κ in `Z` and less than 2κ in `W`; `P_{*ν}`
has degree less than 2κ in `Z` and less than κ in `W`.

**Lemma.** For `z ∈ G^Z_μ` and `α ∈ G^W_ν`,

    P_T(z, W) = P_{μ*}(z, W),        P_T(Z, α) = P_{*ν}(Z, α),
    P_{μ*}(Z, α) = P_{μν}(Z, α),     P_{*ν}(z, W) = P_{μν}(z, W).

*Proof.* By the stitch, `P_T(Z, W) = Σ_μ E^Z_μ(Z) P_{μ*}(Z, W)
= Σ_ν E^W_ν(W) P_{*ν}(Z, W)`. On the parent axes `E^Z_μ` is the indicator
of `G^Z_μ` and `E^W_ν` the indicator of `G^W_ν`, which gives the first two
identities. The same indicator property applied inside the definition of
the partial stitches gives the last two.

## Helper

By the lemma, the stored evaluations of `R_{T,w_h}` on `G^Z_0` are the κ
values `P_{0*}(z, w_h)`, `z ∈ G^Z_0`. The helper interpolates them in `Z`
and evaluates at the requester's `C`, obtaining `P_{0*}(C, w_h)`.
Likewise, the stored evaluations of `L_{T,C_h}` on `G^W_0` are the values
`P_{*0}(C_h, α)`, `α ∈ G^W_0`; the helper interpolates them in `W` and
evaluates at the requester's `w`, obtaining `P_{*0}(C_h, w)`. It returns
both values. Each interpolation is over a roots-of-unity coset and costs
O(κ) field operations.

## Requester

The first values are evaluations of `P_{0*}(C, W)`, a polynomial of degree
less than 2κ in `W`, at the helpers' row labels `w_h`. The requester
decodes this polynomial and evaluates it on `G^W_0` and `G^W_1`. By the
lemma this yields `P_{00}(C, α)` and `P_{01}(C, α)`, that is, `L_{00,C}`
and `L_{01,C}`. Symmetrically, the second values are evaluations of
`P_{*0}(Z, w)`, of degree less than 2κ in `Z`, at the helpers' column
identifiers `C_h`; decoding and evaluating on `G^Z_0` and `G^Z_1` yields
`R_{00,w}` and `R_{10,w}`. The remaining source state follows from the
array formulas (4) and (5):

    L_{1ν,C}[i] = (L_{T,C}[2i+ν] − E^Z_0(C) L_{0ν,C}[i]) / E^Z_1(C),
    R_{μ1,w}[i] = (R_{T,w}[2i+μ] − E^W_0(w) R_{μ0,w}[i]) / E^W_1(w),

for `μ, ν ∈ {0,1}` and `0 ≤ i < κ`. The divisions are defined as in
Proposition 3: `C` and `w` lie off every semantic axis, so the selectors
are nonzero there.

## Cost and conditions

Each helper sends two field elements per stitched segment. With 2κ correct
replies the download is 4κ field elements, the same as in the paper's
Section 4.3. Decoding uses the registry condition for threshold 2κ: the
selected cIDs have distinct column identifiers and distinct row labels,
and their number is at least 2κ plus twice the number of corrupted cIDs
plus the number of missing ones. This procedure returns the two source
primary columns with `μ = 0` and the two source auxiliary rows with
`ν = 0`, which is what a helper's row and column determine directly, where
the procedure of Section 4.3 uses two diagonal segments.
