"""Axiom's exception hierarchy.

Rule 3 (Rules.md) says: never raise on bad *input* -- degrade. These exceptions
are therefore reserved for violations of an internal contract, which are bugs or
corrupted state, never user input.
"""

from __future__ import annotations


class AxiomError(Exception):
    """Base class for every Axiom-raised exception."""


class AxiomContractError(AxiomError):
    """An internal invariant was violated.

    Raised when, for example, a ``VersionManifest`` declares an embedding model
    whose dimensionality disagrees with the loaded FAISS index. Never raised in
    response to a malformed query or an unparseable source file -- those degrade.
    """


class IndexNotFoundError(AxiomError):
    """No index exists at the requested path, or the requested version is unknown."""


class DegradationExhaustedError(AxiomError):
    """Every rung of a degradation ladder failed.

    This is the only case where a degrade path is permitted to raise: there is
    nothing left to fall back to.
    """
