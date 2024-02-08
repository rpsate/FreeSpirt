class Node(object):
    """
    Tree structure
    """

    def __init__(self, kind=None, spelling=None, id=None, parent=None):
        self.id = id
        self.parent = parent
        self.num_children = 0
        self.children = []
        self.kind = kind  #
        self.spelling = spelling  #
        self._depth = -1
        self._width = -1
        self._size = -1

    def add_child(self, child):
        if child:
            child.parent = self
            self.num_children += 1
            self.children.append(child)

    def get_children(self):
        return self.children

    def get_size(self):
        if self._size >= 0:
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].get_size()
        self._size = count
        return self._size

    def get_depth(self):
        if self._depth >= 0:
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].get_depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def get_width(self):
        if self._width >= 0:
            return self._width
        count = self.num_children
        if self.num_children > 0:
            for i in range(self.num_children):
                child_width = self.children[i].get_width()
                if child_width > count:
                    count = child_width
        self._width = count
        return self._width

    def __str__(self):
        children_ids = [child.id for child in self.children]
        return f'<utils.Node> id:{self.id}, kind:{self.kind}, spelling:{self.spelling}, parent id:{self.parent}, children ids:{children_ids}, size: {self.get_size()}, depth: {self.get_depth()}'


