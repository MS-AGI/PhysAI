from .solver import (
    Solver,
    autosolve,
    register_solver,
    register_equation_solver,
    solve_discrete_geometry,
    solve_fipy,
    solve_fenics,
    solve_fenicsx,
    solve_fenicsx_linear,
    solve_meep,
)
from .auto_solver import AutoSolver, AutoSolverConfig, AutoSolverOptimizer

__all__ = [
    "Solver", "autosolve", "AutoSolver", "AutoSolverConfig", "AutoSolverOptimizer",
    "register_solver", "register_equation_solver", "solve_discrete_geometry",
    "solve_fipy", "solve_fenics", "solve_fenicsx", "solve_fenicsx_linear",
    "solve_meep",
]
