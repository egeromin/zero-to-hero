from dataclasses import dataclass
from collections import deque
import math

from graphviz import Digraph


# Goal 1: Implement graph rendering correctly ✅
# Goal 2: Full implementation of operations needed for MLP: + , * and tanh ✅
# Goal 3: Wrappers to build a MLP, to ensure we have all the operations we need.  ✅
#     Goal 3.5: how to test this?
#               test outputs with pytorch.  ✅
# Goal 4: Backprop for all of the engine operations
#     Goal 4.5: Add backprop to MLP wrapper
# Goal 5: Train it on some mock data
# Goal 6: Train on a more complicated an visually appealing dataset


class Value:
    def __init__(
        self,
        data: float | int,
        name: str = "",
        op: str = "",
        children: list["Value"] | None = None,
    ):
        self.data = float(data)
        self.op = op
        self.name = name
        self.grad = 0.0
        self.children = children or []
        # self._backward = lambda: None

    def __repr__(self):
        return "Value(data={}, op={}, name={})".format(self.data, self.op, self.name)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Value(data=other, name="scalar")
        return Value(self.data + other.data, op="+", children=[self, other])

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Value(data=other, name="scalar")
        return Value(self.data * other.data, op="*", children=[self, other])

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        tanh = math.tanh(self.data)  # e^x - 1 / e^x + 1
        return Value(data=tanh, op="tanh", children=[self])


@dataclass(frozen=True, eq=True)
class NodeId:
    from_id: str
    to_id: str

    @classmethod
    def from_value(cls, value: Value) -> "NodeId":
        """
        Return a unique set of IDs identifying the node.
        If the node has an operation, we draw it as two graphviz
        nodes, one with the operation and one with the data.
        When we draw an edge, we always want to draw an arrow
        child.from_id -> parent.to_id

        Having string identifiers for nodes is required by graphviz.
        """
        from_id = str(id(value))
        if value.op:
            to_id = from_id + "-op"
        else:
            to_id = from_id
        return cls(from_id, to_id)


def draw_graph(value: Value):
    dot = Digraph(comment="Operations Graph", strict=True)
    dot.attr(rankdir="LR")

    drawn_nodes = set()
    drawn_edges = set()

    def draw_node(node: Value):
        """Draw the value in graphviz. If it's an op-node, this is two
        nodes, one for the op, connected by an edge to the other one, for the data."""
        node_ids = NodeId.from_value(node)
        if node_ids in drawn_nodes:
            print(f"Node {node} already drawn, skipping")
            drawn_nodes.add(node_ids)
            return

        dot.node(
            name=node_ids.from_id, label=f"{node.name} | {node.data:.4f}", shape="box"
        )
        if node.op:
            dot.node(name=node_ids.to_id, label=node.op)
            dot.edge(node_ids.to_id, node_ids.from_id)

    nodes_to_draw = deque([value])
    while nodes_to_draw:
        # Draw the node and edges to its children.
        # The child nodes may be drawn at a later point in time.
        node = nodes_to_draw.popleft()
        draw_node(node)
        for child in node.children:
            edge = (NodeId.from_value(child).from_id, NodeId.from_value(node).to_id)
            if edge not in drawn_edges:
                dot.edge(*edge)
                drawn_edges.add(edge)

            nodes_to_draw.append(child)

    dot.render("rendered_graph", format="png", cleanup=True, view=True)


def main():
    a = Value(1, "a")
    b = Value(2, "b")
    c = a + b
    c.name = "c"

    d = c + 4
    d.name = "d"

    e = 2 + d
    e.name = "e"

    f = e * 7
    f.name = "f"
    g = 3 * f
    g.name = "g"

    t = g.tanh()
    t.name = "t"

    draw_graph(t)


if __name__ == "__main__":
    main()
