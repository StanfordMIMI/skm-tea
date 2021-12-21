import re
from typing import Sequence, Union


class MarkdownNode:
    """Headings at same level must be uniquely named"""

    def __init__(self, name, content=(), children: Sequence["MarkdownNode"] = None) -> None:
        self.name = name
        self.content = content
        if children is None:
            children = []
        self.children = children

    def add_children(self, children: Union["MarkdownNode", Sequence["MarkdownNode"]]):
        if isinstance(children, MarkdownNode):
            children = [children]
        for c in children:
            assert c not in self.children
            self.children.append(c)

    def remove_children(self, children: Union["MarkdownNode", Sequence["MarkdownNode"]]):
        if isinstance(children, MarkdownNode):
            children = [children]
        for c in children:
            assert c in self.children
        self.children = [c for c in self.children if c not in children]

    def to_dict(self, flatten=False):
        if flatten:
            return self._flattened_dict_repr()
        out = {
            "_name": self.name,
            "content": self.content,
        }
        if self.children:
            out["_children"] = {c.name: c.to_dict() for c in self.children}

    def _flattened_dict_repr(self):
        if not self.children:
            return {self.name: self.content}

        out = {}
        if self.content:
            out[self.name] = self.content
        for c in self.children:
            out.update({f"{self.name}/{k}": v for k, v in c._flattened_dict_repr().items()})
        return out


def parse_markdown(lines: Sequence[str], node: MarkdownNode = None, level: int = 0):
    if node is None:
        node = MarkdownNode("_root", [])

    while len(lines) > 0:
        line = lines[0].strip()

        heading_tag = line.split(" ")[0].strip()
        is_heading = re.match("^#*$", heading_tag) is not None
        if not is_heading:
            node.content.append(line)
            lines = lines[1:]
            continue

        num_heading = len(heading_tag)
        if level >= num_heading:
            return node, lines
        else:
            # import pdb; pdb.set_trace()
            child = MarkdownNode(line.split(" ", maxsplit=1)[1].strip(), [])
            child, lines = parse_markdown(lines[1:], child, level=num_heading)
            node.add_children(child)

    if not node.content and len(node.children) == 1:
        node = node.children[0]
    return node, []
