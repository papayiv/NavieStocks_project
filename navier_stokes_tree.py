"""
Экспрессионные деревья (суперпозиции) для 2D несжимаемых уравнений Навье-Стокса.

Система (в компонентах):
    ∂u/∂t = -(u·∂u/∂x) - (v·∂u/∂y) - (1/ρ)·∂p/∂x + ν·(∂²u/∂x² + ∂²u/∂y²) + g_x
    ∂v/∂t = -(u·∂v/∂x) - (v·∂v/∂y) - (1/ρ)·∂p/∂y + ν·(∂²v/∂x² + ∂²v/∂y²) + g_y

    continuity: ∂u/∂x + ∂v/∂y = 0

Цвета вершин:
    OBSERVED  – наблюдаемые поля (u, v, p)
    PARAMETER – параметры модели (ρ, ν, g_x, g_y)
    LATENT    – операции из словаря (+, -, *, /, d/dx, d/dy, d²/dx², d²/dy²)

Переиспользуемые поддеревья (DAG, не дерево):
    ∂u/∂x  – входит и в конвекцию u-уравнения, и в уравнение неразрывности
    ∂v/∂y  – входит и в конвекцию v-уравнения, и в уравнение неразрывности
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class NodeColor(Enum):
    OBSERVED = "observed"
    LATENT = "latent"
    PARAMETER = "parameter"


@dataclass
class Node:
    id: int
    color: NodeColor
    op: Optional[str] = None
    name: Optional[str] = None
    value: Optional[float] = None
    parents: List["Node"] = field(default_factory=list)

    def __repr__(self):
        if self.color == NodeColor.LATENT:
            return f"V{self.id}[{self.op}]"
        return f"{self.color.value[0].upper()}{self.id}[{self.name}]"


class ExpressionTree:
    """DAG суперпозиции для системы уравнений Навье-Стокса."""

    DICTIONARY = {"+", "-", "*", "/", "d/dx", "d/dy", "d²/dx²", "d²/dy²"}

    def __init__(self):
        self._nodes: List[Node] = []
        self._counter = 0

    def _new_id(self) -> int:
        i = self._counter
        self._counter += 1
        return i

    def observed(self, name: str) -> Node:
        n = Node(id=self._new_id(), color=NodeColor.OBSERVED, name=name)
        self._nodes.append(n)
        return n

    def parameter(self, name: str, value: float) -> Node:
        n = Node(id=self._new_id(), color=NodeColor.PARAMETER, name=name, value=value)
        self._nodes.append(n)
        return n

    def op(self, op: str, *parents: Node) -> Node:
        assert op in self.DICTIONARY, f"операция {op} не в словаре {self.DICTIONARY}"
        n = Node(id=self._new_id(), color=NodeColor.LATENT, op=op, parents=list(parents))
        self._nodes.append(n)
        return n

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    def adjacency_matrix(self) -> np.ndarray:
        """Z[i,j]=1 если узел j является входом узла i."""
        n = len(self._nodes)
        Z = np.zeros((n, n), dtype=float)
        for node in self._nodes:
            for p in node.parents:
                Z[node.id, p.id] = 1.0
        return Z


def build_navier_stokes_2d() -> dict:
    """
    Строит общий DAG для 2D несжимаемых уравнений Навье-Стокса.

    Переиспользуемые узлы:
        du_dx  – в конвекции u-уравнения и в неразрывности
        dv_dy  – в конвекции v-уравнения и в неразрывности
    """
    tree = ExpressionTree()

    # Наблюдаемые переменные
    u = tree.observed("u")
    v = tree.observed("v")
    p = tree.observed("p")

    # Параметры
    rho = tree.parameter("ρ", value=1.0)
    nu = tree.parameter("ν", value=0.01)
    g_x = tree.parameter("g_x", value=0.0)
    g_y = tree.parameter("g_y", value=0.0)
    one = tree.parameter("1", value=1.0)

    # ─── Общие поддеревья (переиспользуемые) ───────────────────────
    du_dx = tree.op("d/dx", u)   # ∂u/∂x  (конвекция u + неразрывность)
    dv_dy = tree.op("d/dy", v)   # ∂v/∂y  (конвекция v + неразрывность)
    du_dy = tree.op("d/dy", u)   # ∂u/∂y
    dv_dx = tree.op("d/dx", v)   # ∂v/∂x

    # ─── u-уравнение: ∂u/∂t = RHS_u ──────────────────────────────
    # конвекция: -(u·∂u/∂x) - (v·∂u/∂y)
    conv_u_x = tree.op("*", u, du_dx)
    conv_u_y = tree.op("*", v, du_dy)
    conv_u = tree.op("+", conv_u_x, conv_u_y)
    neg_conv_u = tree.op("-", conv_u)

    # давление: -(1/ρ)·∂p/∂x
    dp_dx = tree.op("d/dx", p)
    inv_rho = tree.op("/", one, rho)
    press_u = tree.op("*", inv_rho, dp_dx)
    neg_press_u = tree.op("-", press_u)

    # вязкость: ν·(∂²u/∂x² + ∂²u/∂y²)
    d2u_dx2 = tree.op("d²/dx²", u)
    d2u_dy2 = tree.op("d²/dy²", u)
    lap_u = tree.op("+", d2u_dx2, d2u_dy2)
    visc_u = tree.op("*", nu, lap_u)

    # итог: RHS_u = -conv_u - press_u + visc_u + g_x
    sum1_u = tree.op("+", neg_conv_u, neg_press_u)
    sum2_u = tree.op("+", sum1_u, visc_u)
    rhs_u = tree.op("+", sum2_u, g_x)

    # ─── v-уравнение: ∂v/∂t = RHS_v ──────────────────────────────
    # конвекция: -(u·∂v/∂x) - (v·∂v/∂y)
    conv_v_x = tree.op("*", u, dv_dx)
    conv_v_y = tree.op("*", v, dv_dy)
    conv_v = tree.op("+", conv_v_x, conv_v_y)
    neg_conv_v = tree.op("-", conv_v)

    # давление: -(1/ρ)·∂p/∂y
    dp_dy = tree.op("d/dy", p)
    press_v = tree.op("*", inv_rho, dp_dy)   # inv_rho переиспользуется!
    neg_press_v = tree.op("-", press_v)

    # вязкость: ν·(∂²v/∂x² + ∂²v/∂y²)
    d2v_dx2 = tree.op("d²/dx²", v)
    d2v_dy2 = tree.op("d²/dy²", v)
    lap_v = tree.op("+", d2v_dx2, d2v_dy2)
    visc_v = tree.op("*", nu, lap_v)        # nu переиспользуется!

    # итог: RHS_v = -conv_v - press_v + visc_v + g_y
    sum1_v = tree.op("+", neg_conv_v, neg_press_v)
    sum2_v = tree.op("+", sum1_v, visc_v)
    rhs_v = tree.op("+", sum2_v, g_y)

    # ─── Неразрывность: ∂u/∂x + ∂v/∂y = 0 ────────────────────────
    continuity = tree.op("+", du_dx, dv_dy)  # du_dx и dv_dy переиспользуются!

    return {
        "tree": tree,
        "u": u, "v": v, "p": p,
        "params": {"ρ": rho, "ν": nu, "g_x": g_x, "g_y": g_y, "1": one},
        "rhs_u": rhs_u,
        "rhs_v": rhs_v,
        "continuity": continuity,
        "shared_nodes": {"du_dx": du_dx, "dv_dy": dv_dy, "inv_rho": inv_rho, "nu": nu},
    }


def demo():
    built = build_navier_stokes_2d()
    tree = built["tree"]

    print("Вершины DAG Навье-Стокса:")
    for n in tree.nodes:
        print(f"  {repr(n):20s}  color={n.color.value}")

    Z = tree.adjacency_matrix()
    print(f"\nМатрица смежности Z, размер: {Z.shape}")
    print(f"Число рёбер: {int(Z.sum())}")

    # Переиспользуемые узлы
    print("\nПереиспользуемые узлы (входят в несколько родительских операций):")
    for name, node in built["shared_nodes"].items():
        children = [n for n in tree.nodes if node in n.parents]
        print(f"  {name} (id={node.id}) -> {len(children)} потребителей")


if __name__ == "__main__":
    demo()