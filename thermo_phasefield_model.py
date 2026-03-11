from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class ThermoFractureParams:
    """Material/phase-field parameters for the thermo-mechanical mixed-mode model."""

    rho: float
    c_p: float
    k0: float
    kappa: float
    E: float
    nu: float
    alpha: float
    T_ref: float
    G_cI: float
    G_cII: float
    l0: float
    eta_pf: float = 0.0
    eps_r: float = 1e-12


class ThermoMixedModePhaseFieldModel:
    """Implementation of the theoretical model in Sec. 1.1-1.8.

    This class keeps the equations in a compact, reusable form for PINN/FEM workflows.
    All methods support scalar tensors or batched tensors.
    """

    def __init__(self, params: ThermoFractureParams):
        self.p = params

    @property
    def lame(self) -> Tuple[float, float]:
        mu = self.p.E / (2.0 * (1.0 + self.p.nu))
        lam = self.p.E * self.p.nu / ((1.0 + self.p.nu) * (1.0 - 2.0 * self.p.nu))
        return lam, mu

    @property
    def rho_G(self) -> float:
        return self.p.G_cI / self.p.G_cII

    def degradation(self, d: torch.Tensor) -> torch.Tensor:
        # g(d) = (1-d)^2 + kappa
        return (1.0 - d) ** 2 + self.p.kappa

    def heat_conductivity(self, d: torch.Tensor) -> torch.Tensor:
        # k(d) = g(d) * k0
        return self.degradation(d) * self.p.k0

    def elastic_strain(
        self,
        u_xx: torch.Tensor,
        u_yy: torch.Tensor,
        u_xy: torch.Tensor,
        u_yx: torch.Tensor,
        T: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Plane stress form in Eq. (8)-(9)
        thermal = self.p.alpha * (T - self.p.T_ref)
        exx = u_xx - thermal
        eyy = u_yy - thermal
        exy = 0.5 * (u_xy + u_yx)
        ezz = -(self.p.nu / (1.0 - self.p.nu)) * (exx + eyy)
        return {"exx": exx, "eyy": eyy, "exy": exy, "ezz": ezz}

    def split_positive_strain(self, eps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Eq. (14)-(23)
        exx, eyy, exy, ezz = eps["exx"], eps["eyy"], eps["exy"], eps["ezz"]
        em = 0.5 * (exx + eyy)
        ed = 0.5 * (exx - eyy)
        r = torch.sqrt(ed ** 2 + exy ** 2 + self.p.eps_r ** 2)

        e1 = em + r
        e2 = em - r

        e1p = 0.5 * (e1 + torch.abs(e1))
        e2p = 0.5 * (e2 + torch.abs(e2))
        e3p = 0.5 * (ezz + torch.abs(ezz))

        tr = exx + eyy + ezz
        tr_p = 0.5 * (tr + torch.abs(tr))

        chi = ed / r
        eta = exy / r

        exx_p = 0.5 * (e1p + e2p) + 0.5 * (e1p - e2p) * chi
        eyy_p = 0.5 * (e1p + e2p) - 0.5 * (e1p - e2p) * chi
        exy_p = 0.5 * (e1p - e2p) * eta
        ezz_p = e3p

        ep2 = exx_p ** 2 + eyy_p ** 2 + ezz_p ** 2 + 2.0 * exy_p ** 2

        return {
            "tr": tr,
            "tr_p": tr_p,
            "exx_p": exx_p,
            "eyy_p": eyy_p,
            "exy_p": exy_p,
            "ezz_p": ezz_p,
            "ep2": ep2,
        }

    def mode_driving_forces(self, eps_pos: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Eq. (24)-(25)
        lam, mu = self.lame
        psi_I = 0.5 * lam * eps_pos["tr_p"] ** 2
        psi_II = mu * eps_pos["ep2"]
        return {"psi_I": psi_I, "psi_II": psi_II}

    def update_history(
        self,
        H_I_n: torch.Tensor,
        H_II_n: torch.Tensor,
        psi_I: torch.Tensor,
        psi_II: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Eq. (26)-(27), (47)-(48)
        H_I = torch.maximum(H_I_n, psi_I)
        H_II = torch.maximum(H_II_n, psi_II)
        H_e = H_I + self.rho_G * H_II
        return {"H_I": H_I, "H_II": H_II, "H_e": H_e}

    def stress(self, eps: Dict[str, torch.Tensor], eps_pos: Dict[str, torch.Tensor], d: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Eq. (30)-(33)
        lam, mu = self.lame

        tr = eps["exx"] + eps["eyy"] + eps["ezz"]
        sxx0 = lam * tr + 2.0 * mu * eps["exx"]
        syy0 = lam * tr + 2.0 * mu * eps["eyy"]
        sxy0 = 2.0 * mu * eps["exy"]

        sxx_p = lam * eps_pos["tr_p"] + 2.0 * mu * eps_pos["exx_p"]
        syy_p = lam * eps_pos["tr_p"] + 2.0 * mu * eps_pos["eyy_p"]
        sxy_p = 2.0 * mu * eps_pos["exy_p"]

        g = self.degradation(d)

        sxx = sxx0 + (g - 1.0) * sxx_p
        syy = syy0 + (g - 1.0) * syy_p
        sxy = sxy0 + (g - 1.0) * sxy_p

        return {
            "sxx": sxx,
            "syy": syy,
            "sxy": sxy,
            "sxx_pos": sxx_p,
            "syy_pos": syy_p,
            "sxy_pos": sxy_p,
            "sxx_full": sxx0,
            "syy_full": syy0,
            "sxy_full": sxy0,
        }

    def phase_field_coefficients(self, H_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Eq. (42)
        c = torch.as_tensor(self.p.G_cI * self.p.l0, dtype=H_e.dtype, device=H_e.device)
        a = self.p.G_cI / self.p.l0 + 2.0 * H_e
        f = 2.0 * H_e
        d_a = torch.as_tensor(self.p.eta_pf, dtype=H_e.dtype, device=H_e.device)
        return {"c": c, "a": a, "f": f, "d_a": d_a}

    def phase_field_time_form(self, d_dot: torch.Tensor, laplace_d: torch.Tensor, d: torch.Tensor, H_e: torch.Tensor) -> torch.Tensor:
        # Eq. (41)/(49): eta_pf d_dot - G_cI l0 Δd + (G_cI/l0 + 2H_e) d - 2H_e = 0
        return (
            self.p.eta_pf * d_dot
            - self.p.G_cI * self.p.l0 * laplace_d
            + (self.p.G_cI / self.p.l0 + 2.0 * H_e) * d
            - 2.0 * H_e
        )

    def heat_time_form(
        self,
        T_dot: torch.Tensor,
        div_gk_gradT: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        # Eq. (43): rho c_p T_dot - div(g(d)k0 gradT) - Q = 0
        return self.p.rho * self.p.c_p * T_dot - div_gk_gradT - Q
