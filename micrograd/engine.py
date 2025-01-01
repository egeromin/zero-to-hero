import math
import uuid
from collections import deque
from dataclasses import dataclass

from graphviz import Digraph


# Goal 1: Implement graph rendering correctly ✅
# Goal 2: Full implementation of operations needed for MLP: + , * and tanh ✅
# Goal 3: Wrappers to build a MLP, to ensure we have all the operations we need.  ✅
#     Goal 3.5: how to test this?
#               test outputs with pytorch.  ✅
# Goal 4: Backprop
#     Goal 4.1: Backprop for all of the engine operations  ✅
#     Goal 4.2: Add backprop global function  ✅
#     Goal 4.3: Add unit tests for the above  ✅
#     Goal 4.3: Add backprop to MLP wrapper, and unit tests  [LATER, first check how it's used in training]
# Goal 5: Train it on some mock data  ✅
#     Goal 5.1: implement power operation for value ✅
# Goal 6: Train on the sklearn moons dataset
#     Goal 6.1: Try with data loader using mini batches
#     and a different loss function. Try to understand why that particular loss function works better.



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
        self._backward = lambda: None
        self.id = str(uuid.uuid4())

    def __repr__(self):
        return "Value(data={}, op={}, name={}, grad={})".format(
            self.data, self.op, self.name, self.grad
        )

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Value(data=other, name="scalar")

        value = Value(self.data + other.data, op="+", children=[self, other])

        def _backward():
            self.grad += value.grad
            other.grad += value.grad

        value._backward = _backward
        return value

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Value(data=other, name="scalar")

        value = Value(self.data * other.data, op="*", children=[self, other])

        def _backward():
            self.grad += other.data * value.grad
            other.grad += self.data * value.grad

        value._backward = _backward
        return value

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        # Allow only positive integer powers for now, to avoid any issues with complex numbers.
        assert isinstance(other, int) and other >= 1
        value = Value(self.data**other, op="**", children=[self])

        def _backward():
            local_grad = 0.0
            if math.fabs(self.data) > 1e-12:
                # Usual case
                local_grad = other * value.data / self.data
            elif other == 1:
                # identity
                local_grad = 1
            self.grad += local_grad * value.grad

        value._backward = _backward
        return value

    def tanh(self):
        tanh = math.tanh(self.data)  # e^x - e^(-x) / e^x + e^(-x)
        value = Value(data=tanh, op="tanh", children=[self])

        def _backward():
            self.grad += (1 - value.data**2) * value.grad

        value._backward = _backward
        return value

    def backward(self):
        """
        Backprop recursively on the node, its children, and children's children,
        until the whole graph has been backpropped.

        We need to ensure that we have all of the gradient of a given child node
        accumulated, before we do backprop on it.
        """

        # First, topologically sort the nodes.
        # Use depth first search to do this.
        visited = set()
        topological_sort = []

        # Stack size is limited to max depth.
        # TODO: how to refactor in order not to use recursion?
        def visit_recursive(node):
            if node.id not in visited:
                visited.add(node.id)
                for child in node.children:
                    visit_recursive(child)
                topological_sort.append(node)

        visit_recursive(self)
        assert len(visited) == len(topological_sort)
        assert topological_sort[-1] == self
        # Now, topological_sort contains a topological sort of the vertices.

        # Gradient of a variable with respect to itself is 1.
        # This is how we initialise the backprop.
        self.grad = 1.0

        # Traverse the nodes in reverse and backprop
        for node in reversed(topological_sort):
            # print(f"Local backprop on {node}")
            node._backward()


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
            name=node_ids.from_id,
            label=f"{node.name} | data: {node.data:.4f}, grad: {node.grad:.4f}",
            shape="box",
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

    f = e * (1 / 7)
    f.name = "f"
    g = (1 / 3) * f
    g.name = "g"

    t = g.tanh()
    t.name = "t"

    t.backward()

    draw_graph(t)


if __name__ == "__main__":
    main()
