# dixon-e7

Verification programs for the paper

> D. Bhattacharjee, *Dixon–Rosenfeld Planes and Complex E7* (2026).

The paper proves its results by hand. These programs re-check the finite computations.

## Requirements

Python 3 with NumPy and SymPy (`pip install numpy sympy`).

## Running

Run each program from the `scripts` folder, for example:

```
cd scripts
python3 run_sanity.py
```

| Program | What it checks | Paper |
|---|---|---|
| `run_sanity.py` | On the classical algebras T(H, J3(O)), T(O, J3(H)) and T(H, J3(H)), the Jacobi identity forces (α, β) = (1/3, −4) at γ = 1 | Sec. 2 |
| `run_dixon.py` | With full derivations, A1 (dimension 240) and A3 (dimension 259) leave a Jacobi residual for every coefficient choice and every invariant form | Thm. 3.2 |
| `run_variant.py` | The same with R·I removed from A′ (dimensions 214 and 245) | Rem. 3.3 |
| `run_A2.py` | A2 built from O and H3(C⊗H, σ) | Rem. 3.3 |
| `run_lines2.py` | Null spaces for the Dixon–Rosenfeld lines, and for T(H, J2(O)) and T(O, J2(H)) | Thm. 4.1 and the remark after it |
| `run_struct.py` | T(H, J3(O)) and T(O, J3(H)) are compact and simple of dimension 133 and rank 7; T(H, J2(O)) is simple of dimension 66 and rank 6 | Thm. 5.2, Cor. 6.2 |
| `run_invol.py` | The involution θ and its eigenspace dimensions 69/64, 69/64 and 34/32 | Thm. 6.1, Cor. 6.2 |
| `hand.py`, `hand2.py` | The matrix identities in the hand proofs, in exact rational arithmetic | Thms. 3.2, 4.1 |

`alg.py` (algebras, Jordan algebras, derivations) and `tits.py` (Tits' bracket with free coefficients) are shared libraries.

`run_dixon.py`, `run_variant.py` and `run_struct.py` each take a few minutes. The rest finish in seconds.

## License

MIT, see `LICENSE`.
