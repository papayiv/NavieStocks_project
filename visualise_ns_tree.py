"""
Визуализация DAG-структуры уравнений Навье-Стокса.
Три цвета вершин + подсветка переиспользуемых узлов.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D

from navier_stokes_tree import build_navier_stokes_2d, ExpressionTree, NodeColor

COLOR_MAP = {
    NodeColor.OBSERVED: "#4C9AFF",
    NodeColor.PARAMETER: "#FFAB4C",
    NodeColor.LATENT: "#8ED18E",
}

OP_LABEL = {
    "+": "+", "-": "−", "*": "×", "/": "÷",
    "d/dx": "∂x", "d/dy": "∂y",
    "d²/dx²": "∂²x", "d²/dy²": "∂²y",
}


def compute_depths(tree: ExpressionTree) -> dict:
    depth = {}

    def visit(node):
        if node.id in depth:
            return depth[node.id]
        if not node.parents:
            depth[node.id] = 0
        else:
            depth[node.id] = 1 + max(visit(p) for p in node.parents)
        return depth[node.id]

    for n in tree.nodes:
        visit(n)
    return depth


def layout_layered(tree: ExpressionTree) -> dict:
    depth = compute_depths(tree)
    layers = {}
    for n in tree.nodes:
        layers.setdefault(depth[n.id], []).append(n)

    pos = {}
    for d, nodes_in_layer in sorted(layers.items()):
        k = len(nodes_in_layer)
        for i, n in enumerate(nodes_in_layer):
            y = (i - (k - 1) / 2.0) * 1.6
            pos[n.id] = (d * 2.4, y)
    return pos


def node_label(n) -> str:
    if n.color == NodeColor.LATENT:
        return OP_LABEL.get(n.op, n.op)
    if n.color == NodeColor.PARAMETER:
        return f"{n.name}"
    return n.name


def draw_tree(ax, tree, pos, highlight_ids=None, title=""):
    highlight_ids = highlight_ids or set()

    for n in tree.nodes:
        for p in n.parents:
            x1, y1 = pos[p.id]
            x2, y2 = pos[n.id]
            arrow = FancyArrowPatch(
                (x1 + 0.38, y1), (x2 - 0.38, y2),
                arrowstyle="-|>", mutation_scale=10,
                color="#666666", linewidth=1.0, zorder=1,
            )
            ax.add_patch(arrow)

    for n in tree.nodes:
        x, y = pos[n.id]
        is_shared = n.id in highlight_ids
        ec = "#D6336C" if is_shared else "#333333"
        lw = 2.5 if is_shared else 0.8
        r = 0.38 if n.color == NodeColor.LATENT else 0.46
        circle = Circle((x, y), radius=r, facecolor=COLOR_MAP[n.color],
                        edgecolor=ec, linewidth=lw, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, node_label(n), ha="center", va="center",
                fontsize=8, fontweight="bold" if n.color == NodeColor.LATENT else "normal",
                zorder=3)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 1.2, max(ys) + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)


def main():
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    built = build_navier_stokes_2d()
    tree = built["tree"]
    pos = layout_layered(tree)

    shared_ids = {n.id for n in built["shared_nodes"].values()}

    fig, ax = plt.subplots(figsize=(14, 9))
    draw_tree(ax, tree, pos, highlight_ids=shared_ids,
              title="Навье-Стокс 2D: DAG суперпозиции\n"
                    "(розовая обводка — переиспользуемые узлы)")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLOR_MAP[NodeColor.OBSERVED],
               markersize=13, label="наблюдаемая (u, v, p)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLOR_MAP[NodeColor.PARAMETER],
               markersize=13, label="параметр (ρ, ν, g)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLOR_MAP[NodeColor.LATENT],
               markersize=13, label="операция (+,−,×,÷,∂)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="white", markeredgecolor="#D6336C",
               markeredgewidth=2.5, markersize=13,
               label="переиспользуемый узел (shared)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9, frameon=False)

    plt.tight_layout()
    out = output_dir / "navier_stokes_tree_graph.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Сохранено: {out}")

    # Матрица смежности
    Z = tree.adjacency_matrix()
    labels = [node_label(n) for n in tree.nodes]

    fig2, ax2 = plt.subplots(figsize=(9, 8))
    im = ax2.imshow(Z, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=90, fontsize=7)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel("j (вход)")
    ax2.set_ylabel("i (узел)")
    ax2.set_title("Матрица смежности Z: Z[i,j]=1, если j → вход узла i")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out2 = output_dir / "navier_stokes_adjacency.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Сохранено: {out2}")


if __name__ == "__main__":
    main()