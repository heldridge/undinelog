# Reads and Repair on the Product Grid

The paper's Section 3.3 summarizes how a point read, a range read, a
repair, and onboarding use the two interpolation directions of the product
grid, and Proposition 1 states when they return the correct output. This
document gives the mechanics in the paper's notation (the stored slices of
equation (3), the cell points of equation (12)). The code is
`undine/src/coded_array/reconstruct.rs` and `undine/src/unauth/`.

Reads and repair use the same two interpolation directions as in
Borg: a value of a row polynomial comes from primary columns,
and a value of a column polynomial comes from auxiliary rows. Fix one
segment at threshold $\kappa$ with axes $G^Z$ and $G^W$, polynomial $P$,
and stored slices $L_C(W)=P(C,W)$ on $G^W$ and $R_w(Z)=P(Z,w)$ on $G^Z$ as
above, and let helper $h$ have assignments
$(C_h,w_h)$, distinct across a registry. A *point read* of the cell
at $(z_i,\alpha_j)$ takes from each helper its stored value
$L_{C_h}(\alpha_j)=P(C_h,\alpha_j)$, with no arithmetic at the helper;
$\kappa$ correct replies determine $Z\mapsto P(Z,\alpha_j)$, of degree
below $\kappa$, and with it all $\kappa$ cells that share $\alpha_j$. A
*range read* is the symmetric read from the auxiliary rows,
$R_{w_h}(z_i)=P(z_i,w_h)$, which gives the $\kappa$ cells that share
$z_i$; by the cell layout these are $2^\ell$ runs of
$\kappa_{\mathsf{base}}$ consecutive offsets at level $\ell$, so a
contiguous range is served by whole-segment reads for the segments it
covers, $\kappa$ helpers sending $\kappa$ entries each, and by range reads
at its two ends, as in Borg. The *repair* of a primary column for
an identifier $C_\star$ takes $R_{w_h}(C_\star)=P(C_\star,w_h)$ from each
helper, an evaluation off its axis at $O(\kappa)$ field operations since
the axis is a roots-of-unity coset, and interpolates $W\mapsto
P(C_\star,W)$ from $\kappa$ correct replies; the repair of an auxiliary
row is symmetric. Onboarding a new cID runs both repairs for every active
segment, so a fragment of $2\kappa$ entries per segment is obtained with
a download of $2\kappa$ field elements per segment, one per helper and
direction, where Borg's helper returns one row evaluation per auxiliary
row for a column repair.
