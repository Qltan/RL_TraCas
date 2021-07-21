class Vertex(object):
    """
    vertex object
    """
    def __init__(self, key):
        self.key = key
        self.connectedTo = {}    # neighbors of the current vertex

    def addNeighbor(self, nbr, weight):
        self.connectedTo.update({nbr: weight})

    def __str__(self):
        return str(self.key) + '-->' + str([nbr.key for nbr in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.key

    def getWeight(self, nbr):
        weight = self.connectedTo.get(nbr)
        if weight is not None:
            return weight
        else:
            raise KeyError("No such nbr exist!")


class Graph(object):
    """
    graph object
    """
    def __init__(self):
        self.vertexList = {}         # save the information of vertex in the format of {key: Vertex}
        self.vertexNum = 0           # count the vertex numbers of graph
        self.edges = set()
        self.edgeNum = 0
        self.maxNodeNum = 0

    def addVertex(self, key):
        self.vertexList[key] = Vertex(key)
        self.vertexNum += 1
        if key > self.maxNodeNum:
            self.maxNodeNum = key

    def getVertex(self, key):
        if key in self.vertexList.keys():
            vertex = self.vertexList[key]
        else:
            vertex = []
        return vertex

    def __contains__(self, key):
        return key in self.vertexList.values()

    def addEdge(self, f_s, t_e, weight=0):
        f, t = self.getVertex(f_s), self.getVertex(t_e)
        if (f_s, t_e) in self.edges:
            return False
        elif f_s == t_e:
            return False
        else:
            if not f:
                self.addVertex(f_s)
                f = self.getVertex(f_s)
            if not t:
                self.addVertex(t_e)
                t = self.getVertex(t_e)
            f.addNeighbor(t, weight)
            self.edges.add((f_s, t_e))
            self.edgeNum += 1
            return True

    def containEdge(self, source_node, target_node):
        if (source_node, target_node) in self.edges:
            return True
        else:
            return False

    def getVertices(self):
        return self.vertexList.keys()

    def getHop(self, source, target, max_hop):
        current_hop = 1
        s_vertex = self.getVertex(source)
        s_neighbours = s_vertex.getConnections()
        if target in s_neighbours:
            return current_hop
        current_hop += 1
        while current_hop <= max_hop:
            t_neighbours = []
            for v in s_neighbours:
                s_vertex = self.getVertex(v)
                s_neighbours = list(s_vertex.getConnections())
                if target in s_neighbours:
                    return current_hop
                t_neighbours.extend(s_neighbours)
            s_neighbours = t_neighbours
            current_hop += 1
        return -1

    def __iter__(self):
        return iter(self.vertexList.values())


if __name__ == '__main__':
    g = Graph()
    # # Use the set structure for de-duplication
    # t = {(1, 2), (1, 3), (2, 5), (2, 6), (3, 7), (4, 8), (4, 9), (6, 10), (7, 11), (8, 11), (6, 2), (2, 1)}
    # for (s, e) in t:
    #     g.addEdge(s, e)
    # print(g.vertexList.keys())
    # print(g.edgeNum)
    # print(g.edges)
    # d = g.getHop(4, 11, 3)
    # print(g.containEdge(3, 5))
    # print(d)

    g.addEdge(1, 2)
    g.addEdge(3, 2)
    g.addEdge(3, 2)
    print(g.edges)
    print(g.edgeNum)
