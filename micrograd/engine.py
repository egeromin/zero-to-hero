import uuid
from typing import Mapping
from collections import deque, defaultdict

from graphviz import Digraph


# Goal 1: Implement graph rendering correctly
# Goal 2: Full implementation of operations needed for MLP: + , * and tanh
# Goal 3: Backprop for all of the above
# Goal 4: Wrappers to build a MLP
# Goal 5: Train it on some mock data
# Goal 6: Train on a more complicated an visually appealing dataset


class Value:

    def __init__(self, data: float, name: str = "", op: str = "", children: list["Value"] | None = None):
        self.data = data
        self.op = op
        self.name = name
        self.grad = 0.0
        self.children = children or []
        # self._backward = lambda: None

    def __repr__(self):
        return "Value(data={}, op={}, name={})".format(self.data, self.op, self.name)

    def __add__(self, other):
        if isinstance(other, float):
            return Value(self.data + other, op="+")
        return Value(self.data + other.data, op="+", children=[self, other])


def draw_graph(value: Value):
    dot = Digraph(comment="Operations Graph")

    def node_id(value: Value) -> str:
        """
        Return a unique string identifying the node.
        This is required by graphviz
        """
        return str(id(value))

    def draw_node(node: Value):
        dot.node(name=node_id(node), label=f"{node.name} | {node.data:.4f}", shape="box")

    # better to draw a node + edges to its children
    # and then let the children do the same.
    # should work since it's declarative.

    nodes_to_draw = deque([value])
    while nodes_to_draw:
        # Draw the node and edges to its children
        node = nodes_to_draw.popleft()
        draw_node(node)
        for child in node.children:
            dot.edge(node_id(child), node_id(node))
            nodes_to_draw.append(child)

    dot.render("rendered_graph", format="png", cleanup=True, view=True)


def main():
    a = Value(1, "a")
    b = Value(2, "b")
    c = a + b
    c.name = "c"
    print(c)
    draw_graph(c)


if __name__ == "__main__":
    main()
