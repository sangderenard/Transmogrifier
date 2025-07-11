#!/usr/bin/env python3
"""
==========================================================
| DEC.py                                                 |
| Canonical symbolic Discrete Exterior Calculus (DEC)    |
| ----------------------------------------------------- |
| Pure symbolic textbook operators, implemented in sympy |
| for rigorous differential geometry on discrete spaces.|
==========================================================

References:
- Hirani, "Discrete Exterior Calculus"
- Desbrun et al., "Discrete Differential Forms for Computational Modeling"
- Abraham, Marsden, Ratiu, "Manifolds, Tensor Analysis, and Applications"
- Marsden & Hughes, "Mathematical Foundations of Elasticity"
"""

import sympy as sp

# ------------------------------------------------------
# DEC symbolic fundamental forms & standard symbols
# ------------------------------------------------------

class DEC:
    """
    Canonical symbolic Discrete Exterior Calculus operators.
    Provides symbolic definitions for:
      - 0-forms (scalar fields at vertices)
      - 1-forms (integrals along edges)
      - 2-forms (fluxes through faces)
      - Exterior derivative d
      - Hodge star ⋆
      - Codifferential δ
      - Laplace-de Rham Δ
    """

    # ------------------------------------------------------
    # Standard symbolic indices for discrete geometry
    # ------------------------------------------------------
    i, j, k = sp.symbols('i j k', integer=True)
    dx, dy, dz = sp.symbols('dx dy dz', real=True)
    ε = sp.Symbol('ε', real=True)  # small parameter for expansions

    # ------------------------------------------------------
    # 0-form: scalar field on vertices
    # ------------------------------------------------------
    @staticmethod
    def zero_form():
        """
        φ(i) : 0-form, scalar field defined on vertices.
        """
        φ = sp.Function('φ')
        return φ(DEC.i)

    # ------------------------------------------------------
    # 1-form: integrals over edges
    # ------------------------------------------------------
    @staticmethod
    def one_form():
        """
        ω(i, j) : 1-form, integral along edge (i,j).
        """
        ω = sp.Function('ω')
        return ω(DEC.i, DEC.j)

    # ------------------------------------------------------
    # 2-form: integrals over faces
    # ------------------------------------------------------
    @staticmethod
    def two_form():
        """
        σ(i, j, k) : 2-form, integral over face (i,j,k).
        """
        σ = sp.Function('σ')
        return σ(DEC.i, DEC.j, DEC.k)

    # ------------------------------------------------------
    # Exterior derivative d
    # ------------------------------------------------------
    @staticmethod
    def exterior_derivative_0_to_1(φ):
        """
        Discrete exterior derivative of 0-form to 1-form:
            (dφ)(i, j) = φ(j) - φ(i)
        """
        return φ(DEC.j) - φ(DEC.i)

    @staticmethod
    def exterior_derivative_1_to_2(ω):
        """
        Discrete exterior derivative of 1-form to 2-form:
            (dω)(i, j, k) = ω(j,k) - ω(i,k) + ω(i,j)
        """
        return ω(DEC.j, DEC.k) - ω(DEC.i, DEC.k) + ω(DEC.i, DEC.j)

    # ------------------------------------------------------
    # Hodge star ⋆
    # ------------------------------------------------------
    @staticmethod
    def hodge_star_0_to_n(φ, vol_dual_cell):
        """
        Hodge star on 0-form to n-form on dual cell:
            ⋆φ(i) = φ(i) * vol_dual_cell(i)
        """
        return φ(DEC.i) * vol_dual_cell(DEC.i)

    @staticmethod
    def hodge_star_1_to_n_minus_1(ω, vol_dual_edge):
        """
        Hodge star on 1-form to (n-1)-form on dual edge:
            ⋆ω(i, j) = ω(i, j) * vol_dual_edge(i, j)
        """
        return ω(DEC.i, DEC.j) * vol_dual_edge(DEC.i, DEC.j)

    # ------------------------------------------------------
    # Codifferential δ
    # ------------------------------------------------------
    @staticmethod
    def codifferential(star_ω_expr):
        """
        Codifferential δ on 1-form:
            δ = ⋆^{-1} d ⋆
        In discrete DEC, implemented combinatorially as:
            (δω)(i) = Σ_j [⋆ω(i,j) - ⋆ω(j,i)]
        """
        return star_ω_expr - star_ω_expr.subs({DEC.i: DEC.j, DEC.j: DEC.i})

    # ------------------------------------------------------
    # Laplace-de Rham Δ
    # ------------------------------------------------------
    @staticmethod
    def laplace_de_rham(φ, vol_dual_cell, vol_dual_edge):
        """
        Laplace-de Rham operator on scalar field (0-form):
            Δφ = δ d φ
        Computes symbolically:
            dφ = exterior_derivative_0_to_1(φ)
            ⋆dφ = hodge_star_1_to_n_minus_1(dφ)
            δ⋆dφ = codifferential(⋆dφ)
        """
        dφ = DEC.exterior_derivative_0_to_1(φ)
        star_dφ = DEC.hodge_star_1_to_n_minus_1(lambda i, j: dφ.subs({DEC.i: i, DEC.j: j}), vol_dual_edge)
        δdφ = DEC.codifferential(star_dφ)
        return δdφ

    # ------------------------------------------------------
    # Integration operators
    # ------------------------------------------------------
    @staticmethod
    def integrate_over_vertex(φ, volume_weight):
        """
        ∫_dual_cell φ = φ(i) * volume_weight(i)
        """
        return φ(DEC.i) * volume_weight(DEC.i)

    @staticmethod
    def integrate_over_edge(ω, length_weight):
        """
        ∫_edge ω = ω(i,j) * length_weight(i,j)
        """
        return ω(DEC.i, DEC.j) * length_weight(DEC.i, DEC.j)

    @staticmethod
    def integrate_over_face(σ, area_weight):
        """
        ∫_face σ = σ(i,j,k) * area_weight(i,j,k)
        """
        return σ(DEC.i, DEC.j, DEC.k) * area_weight(DEC.i, DEC.j, DEC.k)


# ------------------------------------------------------
# Example symbolic DEC demonstration
# ------------------------------------------------------

def demo():
    print("\n=== Canonical DEC Symbolic Demo ===")

    φ = sp.Function('φ')
    ω = sp.Function('ω')
    σ = sp.Function('σ')
    vol_dual_cell = sp.Function('vol_dual_cell')
    vol_dual_edge = sp.Function('vol_dual_edge')

    print("\nExterior derivative dφ from 0-form to 1-form:")
    print(sp.pretty(DEC.exterior_derivative_0_to_1(φ), use_unicode=True))

    print("\nExterior derivative dω from 1-form to 2-form:")
    print(sp.pretty(DEC.exterior_derivative_1_to_2(ω), use_unicode=True))

    print("\nHodge star on 0-form to n-form dual:")
    print(sp.pretty(DEC.hodge_star_0_to_n(φ, vol_dual_cell), use_unicode=True))

    print("\nLaplace-de Rham Δφ:")
    Δφ = DEC.laplace_de_rham(φ, vol_dual_cell, vol_dual_edge)
    print(sp.pretty(Δφ, use_unicode=True))


if __name__ == "__main__":
    demo()
