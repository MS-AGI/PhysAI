"""Compile a safe, useful subset of LaTeX PDE equations into residuals.

Supported notation includes arithmetic, scalar fields, partial derivative
fractions and subscripts, Laplacians, and common scalar functions. Expressions
are checked with Python's AST and interpreted directly; user text is never
passed to eval.
"""
from __future__ import annotations

import ast
import keyword
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from physai.core.pde_residual import PDEResidual, register_pde


_GREEK = {
    r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma",
    r"\delta": "delta", r"\epsilon": "epsilon", r"\theta": "theta",
    r"\kappa": "kappa", r"\lambda": "lambda", r"\mu": "mu",
    r"\nu": "nu", r"\omega": "omega", r"\rho": "rho",
    r"\sigma": "sigma", r"\tau": "tau", r"\phi": "phi",
    r"\psi": "psi", r"\xi": "xi", r"\chi": "chi",
}
_LATEX_FUNCTIONS = {
    r"\sin": "sin", r"\cos": "cos", r"\tan": "tan",
    r"\tanh": "tanh", r"\exp": "exp", r"\log": "log",
    r"\ln": "log", r"\sqrt": "sqrt",
}
_SAFE_FUNCTIONS = {"sin", "cos", "tan", "tanh", "exp", "log", "sqrt", "abs"}
_SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
)


@dataclass(frozen=True)
class _ParsedEquation:
    source: str
    lhs: ast.Expression
    rhs: ast.Expression
    derivative_order: int
    nonlinear: bool


def _read_group(text: str, start: int) -> Tuple[str, int]:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        raise ValueError("Unexpected end of LaTeX after a command.")
    if text[start] != "{":
        if text[start] == "\\":
            match = re.match(r"\\[A-Za-z]+", text[start:])
            if match:
                return match.group(0), start + len(match.group(0))
        match = re.match(r"[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?|.", text[start:])
        if not match:
            raise ValueError("Expected a LaTeX argument.")
        return match.group(0), start + len(match.group(0))
    depth, pos = 1, start + 1
    while pos < len(text) and depth:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth:
        raise ValueError("Unclosed '{' group in LaTeX equation.")
    return text[start + 1:pos - 1], pos


def _latex_symbol(token: str) -> str:
    token = re.sub(r"\\(?:mathrm|mathit|mathbf)\s*\{([^{}]+)\}", r"\1", token.strip())
    token = re.sub(r"\s+", "", token)
    applied = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\([^()]*\)", token)
    return (applied.group(1) if applied else token).lstrip("\\")


def _replace_derivative_fractions(text: str, fields: Sequence[str], coordinates: Sequence[str]) -> str:
    commands = (r"\frac", r"\dfrac", r"\tfrac")
    pos = 0
    while True:
        found = [(text.find(command, pos), command) for command in commands]
        found = [(idx, command) for idx, command in found if idx >= 0]
        if not found:
            break
        idx, command = min(found)
        numerator_text, after_num = _read_group(text, idx + len(command))
        denominator_text, after_den = _read_group(text, after_num)
        numerator_text = _replace_derivative_fractions(numerator_text, fields, coordinates)
        denominator_text = _replace_derivative_fractions(denominator_text, fields, coordinates)
        numerator = re.fullmatch(
            r"\s*\\partial(?:\s*\^\s*\{?(\d+)\}?)?\s*(.+?)\s*",
            numerator_text,
        )
        variables = re.findall(
            r"\\partial(?:\s*_\s*)?\{?([A-Za-z][A-Za-z0-9_]*)\}?"
            r"(?:\s*\^\s*\{?(\d+)\}?)?",
            denominator_text,
        )
        replacement = f"({numerator_text})/({denominator_text})"
        if numerator and variables:
            field = _latex_symbol(numerator.group(2))
            derivative_vars = []
            valid = field in fields
            for variable, power in variables:
                variable = _latex_symbol(variable)
                if variable not in coordinates:
                    valid = False
                    break
                derivative_vars.extend([variable] * int(power or 1))
            if valid and len(derivative_vars) == int(numerator.group(1) or 1):
                replacement = "D(" + ", ".join([field, *derivative_vars]) + ")"
        text = text[:idx] + replacement + text[after_den:]
        pos = idx + len(replacement)
    return text


def _replace_shorthand_derivatives(text: str, fields: Sequence[str], coordinates: Sequence[str]) -> str:
    for field in sorted(fields, key=len, reverse=True):
        f = re.escape(field)
        for pattern in (
            rf"\\partial\s*_\s*\{{([A-Za-z]+)\}}\s*{f}\b",
            rf"\\partial\s*_\s*([A-Za-z]+)\s*{f}\b",
        ):
            text = re.sub(
                pattern,
                lambda m: "D(" + ", ".join([field, *list(m.group(1))]) + ")",
                text,
            )
        for pattern in (rf"\b{f}_\{{([A-Za-z]+)\}}", rf"\b{f}_([A-Za-z]+)\b"):
            text = re.sub(
                pattern,
                lambda m: "D(" + ", ".join([field, *list(m.group(1))]) + ")"
                if all(axis in coordinates for axis in m.group(1)) else m.group(0),
                text,
            )
    text = re.sub(r"\\(?:nabla\s*\^\s*\{?2\}?|Delta|triangle)", "Lap", text)
    for field in sorted(fields, key=len, reverse=True):
        text = re.sub(rf"\bLap\s*{re.escape(field)}\b", f"Lap({field})", text)
    return text


def _latex_to_python(expression: str, fields: Sequence[str], coordinates: Sequence[str]) -> str:
    text = expression.strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = text.replace(r"\left", "").replace(r"\right", "")
    for spacing in (r"\,", r"\;", r"\quad", r"\qquad", r"\!", r"\ "):
        text = text.replace(spacing, " ")
    text = _replace_derivative_fractions(text, fields, coordinates)
    text = _replace_shorthand_derivatives(text, fields, coordinates)
    coordinate_args = r"\s*,\s*".join(re.escape(axis) for axis in coordinates)
    for field in sorted(fields, key=len, reverse=True):
        text = re.sub(
            rf"\b{re.escape(field)}\s*\(\s*{coordinate_args}\s*\)",
            field,
            text,
        )
    for macro, name in _LATEX_FUNCTIONS.items():
        text = text.replace(macro, name)
    for macro, name in _GREEK.items():
        text = text.replace(macro, name)
    text = text.replace(r"\pi", "pi")
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    while "sqrt{" in text:
        idx = text.index("sqrt{") + len("sqrt")
        group, end = _read_group(text, idx)
        text = text[:idx - len("sqrt")] + f"sqrt({group})" + text[end:]
    text = text.replace("^", "**").replace("{", "(").replace("}", ")")
    # Juxtaposition such as u u_x denotes multiplication in PDE notation.
    text = re.sub(r"(?<=[A-Za-z0-9_)])\s+(?=[A-Za-z_(])", "*", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z_(])", "*", text)
    for field in sorted(fields, key=len, reverse=True):
        text = re.sub(rf"\b({re.escape(field)})(?=(?:D|Lap)\()", r"\1*", text)
    return text.strip()


def _validate_ast(
    expression: str, allowed_names: set, source: str,
    fields: Sequence[str], coordinates: Sequence[str],
) -> ast.Expression:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Invalid LaTeX PDE syntax in {source!r}: {exc.msg} near column {exc.offset}. "
            f"Parsed expression: {expression!r}."
        ) from None
    for node in ast.walk(parsed):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"Unsupported syntax in PDE expression {source!r}: {type(node).__name__}.")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unknown symbol {node.id!r} in PDE expression {source!r}.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS | {"D", "Lap"}:
                raise ValueError(f"Unsupported function call in PDE expression {source!r}.")
            if node.keywords:
                raise ValueError("Keyword arguments are not allowed in a PDE expression.")
            if node.func.id == "D":
                if (len(node.args) < 2 or not isinstance(node.args[0], ast.Name)
                        or node.args[0].id not in fields):
                    raise ValueError("D requires a field followed by one or more coordinate names.")
                if any(not isinstance(arg, ast.Name) or arg.id not in coordinates for arg in node.args[1:]):
                    raise ValueError("D derivative variables must be declared coordinate names.")
            elif node.func.id == "Lap":
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Name) or node.args[0].id not in fields:
                    raise ValueError("Lap takes one field name.")
            elif len(node.args) != 1:
                raise ValueError(f"Function {node.func.id} takes one argument.")
    return parsed


def _has_nonlinear_product(node: ast.AST, fields: set) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
        if any({n.id for n in ast.walk(arg) if isinstance(n, ast.Name)} & fields for arg in node.args):
            return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Pow)):
        left = {n.id for n in ast.walk(node.left) if isinstance(n, ast.Name)} & fields
        right = {n.id for n in ast.walk(node.right) if isinstance(n, ast.Name)} & fields
        if (left and right) or (isinstance(node.op, ast.Pow) and left):
            return True
    return any(_has_nonlinear_product(child, fields) for child in ast.iter_child_nodes(node))


def _parse_equations(
    equations: str | Sequence[str],
    fields: Sequence[str],
    coordinates: Sequence[str],
    parameters: Mapping[str, Any],
) -> Tuple[_ParsedEquation, ...]:
    sources = (equations,) if isinstance(equations, str) else tuple(equations)
    if not sources or any(not isinstance(eq, str) or not eq.strip() for eq in sources):
        raise ValueError("equations must be a non-empty LaTeX equation or sequence of equations.")
    fields = (fields,) if isinstance(fields, str) else tuple(fields)
    coordinates = (coordinates,) if isinstance(coordinates, str) else tuple(coordinates)
    for names, label in ((fields, "fields"), (coordinates, "coordinates")):
        if not names or len(set(names)) != len(names):
            raise ValueError(f"{label} must contain unique, non-empty names.")
        if any(
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
            or keyword.iskeyword(name)
            for name in names
        ):
            raise ValueError(f"{label} names must be Python-style identifiers.")
    if set(fields) & set(coordinates):
        raise ValueError("Field names and coordinate names must be distinct.")
    if set(parameters) & (set(fields) | set(coordinates)):
        raise ValueError("Parameter names cannot overlap fields or coordinates.")
    for name, value in parameters.items():
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
                or keyword.iskeyword(name)):
            raise ValueError(f"Invalid parameter name {name!r}.")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Parameter {name!r} must be a finite real number.")

    allowed = set(fields) | set(coordinates) | set(parameters) | {"pi", "e"}
    allowed |= _SAFE_FUNCTIONS | {"D", "Lap"}
    parsed = []
    for source in sources:
        normalized = re.sub(
            r"\\begin\{(?:equation|aligned|split)\}|\\end\{(?:equation|aligned|split)\}",
            "", source,
        ).replace(r"\[", "").replace(r"\]", "")
        if normalized.count("=") != 1:
            raise ValueError(f"Each PDE equation must contain exactly one '=': {source!r}.")
        lhs_source, rhs_source = normalized.split("=", 1)
        lhs = _validate_ast(_latex_to_python(lhs_source, fields, coordinates), allowed, source, fields, coordinates)
        rhs = _validate_ast(_latex_to_python(rhs_source, fields, coordinates), allowed, source, fields, coordinates)
        derivatives = [
            node for root in (lhs, rhs) for node in ast.walk(root)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {"D", "Lap"}
        ]
        order = max(1, max(
            (len(node.args) - 1 if node.func.id == "D" else 2 for node in derivatives),
            default=0,
        ))
        nonlinear = _has_nonlinear_product(lhs, set(fields)) or _has_nonlinear_product(rhs, set(fields))
        parsed.append(_ParsedEquation(source, lhs, rhs, order, nonlinear))
    return tuple(parsed)


class LatexPDEResidual(PDEResidual):
    """Backend-differentiable residual compiled from LaTeX equations."""

    def __init__(
        self,
        backend,
        equations: str | Sequence[str],
        *,
        fields: Sequence[str] = ("u",),
        coordinates: Sequence[str] = ("x", "t"),
        parameters: Optional[Mapping[str, Any]] = None,
        diff_mode: str = "reverse",
    ) -> None:
        super().__init__(backend, diff_mode=diff_mode)
        self.fields = (fields,) if isinstance(fields, str) else tuple(fields)
        self.coordinates = (coordinates,) if isinstance(coordinates, str) else tuple(coordinates)
        self.parameters = dict(parameters or {})
        if any(not isinstance(v, (int, float)) or not math.isfinite(float(v))
               for v in self.parameters.values()):
            raise ValueError("LaTeX residual parameters must be finite real numbers.")
        self.equations = _parse_equations(equations, self.fields, self.coordinates, self.parameters)
        self.n_components = len(self.fields)
        self.output_components = len(self.equations)
        self.order = max(eq.derivative_order for eq in self.equations)
        self.nonlinear = any(eq.nonlinear for eq in self.equations)

    def _differentiate(self, model_fn, points, field: str, axes: Sequence[str]):
        backend = self.backend
        field_index = self.fields.index(field)

        def current(query):
            prediction = model_fn(query)
            if len(self.fields) == 1 and prediction.ndim == 1:
                return prediction
            if prediction.shape[-1] < len(self.fields):
                raise ValueError(
                    f"model_fn returned {prediction.shape[-1]} fields, but {len(self.fields)} were declared."
                )
            return prediction[..., field_index]

        for axis in (self.coordinates.index(name) for name in axes):
            previous = current
            gradient = backend.grad(lambda query: backend.sum(previous(query)), mode=self.diff_mode)
            current = lambda query, grad=gradient, index=axis: grad(query)[..., index]
        return current(points)

    def _evaluate(self, node, model_fn, points):
        backend = self.backend
        if isinstance(node, ast.Expression):
            return self._evaluate(node.body, model_fn, points)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are supported.")
        if isinstance(node, ast.Name):
            if node.id in self.fields:
                prediction = model_fn(points)
                if len(self.fields) == 1 and prediction.ndim == 1:
                    return prediction
                if prediction.shape[-1] < len(self.fields):
                    raise ValueError(
                        f"model_fn returned {prediction.shape[-1]} fields, but {len(self.fields)} were declared."
                    )
                return prediction[..., self.fields.index(node.id)]
            if node.id in self.coordinates:
                return points[..., self.coordinates.index(node.id)]
            if node.id in self.parameters:
                return self.parameters[node.id]
            if node.id == "pi": return math.pi
            if node.id == "e": return math.e
            raise ValueError(f"Unknown PDE symbol {node.id!r}.")
        if isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand, model_fn, points)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            lhs = self._evaluate(node.left, model_fn, points)
            rhs = self._evaluate(node.right, model_fn, points)
            if isinstance(node.op, ast.Add): return lhs + rhs
            if isinstance(node.op, ast.Sub): return lhs - rhs
            if isinstance(node.op, ast.Mult): return lhs * rhs
            if isinstance(node.op, ast.Div): return lhs / rhs
            if isinstance(node.op, ast.Pow): return lhs ** rhs
            if isinstance(node.op, ast.Mod): return lhs % rhs
        if isinstance(node, ast.Call):
            name = node.func.id
            if name in {"D", "Lap"}:
                if not node.args or not isinstance(node.args[0], ast.Name) or node.args[0].id not in self.fields:
                    raise ValueError(f"{name} expects a declared field as its first argument.")
                field = node.args[0].id
                if name == "D":
                    if len(node.args) < 2 or any(
                        not isinstance(arg, ast.Name) or arg.id not in self.coordinates for arg in node.args[1:]
                    ):
                        raise ValueError("D(field, coordinate, ...) requires declared coordinate names.")
                    return self._differentiate(model_fn, points, field, [arg.id for arg in node.args[1:]])
                if len(node.args) != 1:
                    raise ValueError("Lap takes exactly one field argument.")
                spatial = [axis for axis in self.coordinates if axis != "t"]
                if not spatial:
                    raise ValueError("Lap requires at least one non-time coordinate.")
                terms = [self._differentiate(model_fn, points, field, (axis, axis)) for axis in spatial]
                result = terms[0]
                for term in terms[1:]:
                    result = result + term
                return result
            args = [self._evaluate(arg, model_fn, points) for arg in node.args]
            if len(args) != 1:
                raise ValueError(f"Function {name} takes one argument.")
            return backend.abs(args[0]) if name == "abs" else getattr(backend, name)(args[0])
        raise ValueError(f"Unsupported PDE expression node {type(node).__name__}.")

    def __call__(self, model_fn, points):
        if points.shape[-1] != len(self.coordinates):
            raise ValueError(
                f"Equation declares {len(self.coordinates)} coordinates; received {points.shape[-1]} columns."
            )
        residuals = [
            self._evaluate(eq.lhs, model_fn, points) - self._evaluate(eq.rhs, model_fn, points)
            for eq in self.equations
        ]
        return residuals[0] if len(residuals) == 1 else self.backend.stack(residuals, axis=-1)


def build_latex_residual(
    equations: str | Sequence[str],
    backend,
    *,
    fields: Sequence[str] = ("u",),
    coordinates: Sequence[str] = ("x", "t"),
    parameters: Optional[Mapping[str, Any]] = None,
    diff_mode: str = "reverse",
) -> LatexPDEResidual:
    """Validate and compile equations, e.g. a time-dependent Burgers PDE."""
    return LatexPDEResidual(
        backend, equations, fields=fields, coordinates=coordinates,
        parameters=parameters, diff_mode=diff_mode,
    )


def register_latex_pde(
    name: str,
    equations: str | Sequence[str],
    *,
    fields: Sequence[str] = ("u",),
    coordinates: Sequence[str] = ("x", "t"),
    parameters: Optional[Mapping[str, Any]] = None,
    meta: Any = None,
    aliases: Sequence[str] = (),
    overwrite: bool = False,
):
    """Validate, compile, and register equations for build_residual(name, backend)."""
    fields = (fields,) if isinstance(fields, str) else tuple(fields)
    coordinates = (coordinates,) if isinstance(coordinates, str) else tuple(coordinates)
    defaults = dict(parameters or {})
    sources = (equations,) if isinstance(equations, str) else tuple(equations)
    parsed = _parse_equations(sources, fields, coordinates, defaults)

    class RegisteredLatexResidual(LatexPDEResidual):
        def __init__(self, backend, *, diff_mode="reverse", **overrides):
            unknown = set(overrides) - set(defaults)
            if unknown:
                raise TypeError(f"Unknown PDE parameter(s): {sorted(unknown)}.")
            super().__init__(
                backend, sources, fields=fields, coordinates=coordinates,
                parameters={**defaults, **overrides}, diff_mode=diff_mode,
            )

    RegisteredLatexResidual.__name__ = "RegisteredLatexResidual"
    RegisteredLatexResidual.__qualname__ = f"{name}_LatexResidual"
    if meta is None:
        meta = {
            "order": max(eq.derivative_order for eq in parsed),
            "nonlinear": any(eq.nonlinear for eq in parsed),
            "n_components": len(fields),
        }
    return register_pde(
        name, RegisteredLatexResidual, meta=meta, aliases=aliases, overwrite=overwrite,
    )


__all__ = ["LatexPDEResidual", "build_latex_residual", "register_latex_pde"]
