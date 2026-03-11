# phase-field-fracture-with-pidl



## Code associated with the paper "[Phase-field modeling of fracture with physics-informed deep learning](https://www.sciencedirect.com/science/article/pii/S0045782524003608)"

### Source files are located in the source directory.
### Files corresponding to examples of crack nucleation, propagation, kinking, branching, and coalescense are located in the examples directory.
## Thermo-mechanical mixed-mode model (new)

A reusable implementation of the theoretical model shown in the provided equations is added in:

- `thermo_phasefield_model.py`

Core coverage:

- Heat conduction degradation: `g(d)=(1-d)^2+κ`, `k(d)=g(d)k0`
- Plane-stress thermo-elastic strain and Lamé constants
- Mixed-mode strain split and driving forces (`ψ_I`, `ψ_II`)
- Dual history variables (`H_I`, `H_II`) and mixed driving force (`H_e`)
- Positive-stress degradation coupling `σ = σ0 + (g(d)-1)σ+`
- Phase-field PDE coefficient form and time-continuous residual form
