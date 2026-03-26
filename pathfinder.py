import math
import heapq
import networkx as nx

class DStarLite:
    INF = float("inf")

    def __init__(self, graph, start_node, goal_node):
        self.graph = graph
        self.start = start_node
        self.goal = goal_node
        self.k_m = 0.0
        self.g = {}
        self.rhs = {}
        self._heap = []
        self._in_heap = set()
        self.initialize()

    def heuristic(self, a, b):
        try:
            ax = self.graph.nodes[a].get('x', 0)
            ay = self.graph.nodes[a].get('y', 0)
            bx = self.graph.nodes[b].get('x', 0)
            by = self.graph.nodes[b].get('y', 0)
            return math.sqrt((ax-bx)**2 + (ay-by)**2)
        except (KeyError, TypeError):
            return 0.0

    def edge_cost(self, u, v):
        if not self.graph.has_node(u) or not self.graph.has_node(v):
            return self.INF
        if self.graph.has_edge(u, v):
            edge_data = self.graph.get_edge_data(u, v)
            if isinstance(edge_data, dict):
                costs = []
                for key, data in edge_data.items():
                    costs.append(data.get('length', 1.0))
                return min(costs)
            return 1.0
        return self.INF

    def predecessors(self, node):
        try:
            return list(self.graph.neighbors(node))
        except nx.NetworkXError:
            return []

    def calculate_key(self, node):
        g_val = self.g.get(node, self.INF)
        rhs_val = self.rhs.get(node, self.INF)
        min_val = min(g_val, rhs_val)
        k1 = min_val + self.heuristic(self.start, node) + self.k_m
        k2 = min_val
        return k1,k2

    def heap_push(self, node):
        key = self.calculate_key(node)
        heapq.heappush(self._heap, (key, node))
        self._in_heap.add(node)

    def heap_top_key(self):
        while self._heap:
            key, node = self._heap[0]
            curr_key = self.calculate_key(node)
            if key == curr_key and node in self._in_heap:
                return key
            heapq.heappop(self._heap)
            if node in self._in_heap:
                pass
        return (self.INF, self.INF)

    def heap_pop(self):
        while self._heap:
            key, node = heapq.heappop(self._heap)
            if node in self._in_heap:
                curr_key = self.calculate_key(node)
                if key == curr_key:
                    self._in_heap.discard(node)
                    return key, node
                else:
                    heapq.heappush(self._heap, (curr_key, node))
        return None, None

    def initialize(self):
        self.g.clear()
        self.rhs.clear()
        self._heap.clear()
        self._in_heap.clear()
        self.k_m = 0.0

        if not self.graph.has_node(self.goal) or not self.graph.has_node(self.start):
            return

        self.rhs[self.goal] = 0.0
        self.heap_push(self.goal)

    def update_vertex(self, node):
        if node == self.goal:
            return
        if not self.graph.has_node(node):
            return

        min_cost = self.INF
        for neighbor in self.predecessors(node):
            c = self.edge_cost(node, neighbor)
            g_n = self.g.get(neighbor, self.INF)
            if c + g_n < min_cost:
                min_cost = c + g_n
        self.rhs[node] = min_cost

        self._in_heap.discard(node)

        if self.g.get(node, self.INF) != self.rhs.get(node, self.INF):
            self.heap_push(node)

    def compute_shortest_path(self):
        if not self.graph.has_node(self.goal) or not self.graph.has_node(self.start):
            return
        
        max_iterations = len(self.graph.nodes) * 2
        iterations = 0
        
        start_key = self.calculate_key(self.start)
        g_start = self.g.get(self.start, self.INF)
        rhs_start = self.rhs.get(self.start, self.INF)
        while (self.heap_top_key() < start_key or rhs_start != g_start) and iterations < max_iterations:
            iterations += 1
            top_key = self.heap_top_key()
            key, node = self.heap_pop()
            if node is None:
                break
            
            new_key = self.calculate_key(node)
            if top_key < new_key:
                self.heap_push(node)
            elif self.g.get(node, self.INF) > self.rhs.get(node, self.INF):
                self.g[node] = self.rhs.get(node, self.INF)
                for pred in self.predecessors(node):
                    self.update_vertex(pred)
            else:
                self.g[node] = self.INF
                self.update_vertex(node)
                for pred in self.predecessors(node):
                    self.update_vertex(pred)

            start_key = self.calculate_key(self.start)
            g_start = self.g.get(self.start, self.INF)
            rhs_start = self.rhs.get(self.start, self.INF)

    def get_path(self):
        if self.g.get(self.start, self.INF) == self.INF:
            return []

        path = [self.start]
        current = self.start
        max_steps = len(self.graph.nodes)
        steps = 0

        while current != self.goal and steps < max_steps:
            steps += 1
            neighbors = self.predecessors(current)
            if not neighbors:
                return []

            best_next = None
            best_cost = self.INF
            for n in neighbors:
                c = self.edge_cost(current, n)
                g_n = self.g.get(n, self.INF)
                total = c + g_n
                if total< best_cost:
                    best_cost = total
                    best_next = n
            if best_next is None or best_cost == self.INF:
                return []

            if best_next in path:
                return []

            path.append(best_next)
            current = best_next

        if current != self.goal:
            return []
        return path

    def notify_edge_changed(self, u, v):
        self.update_vertex(u)
        self.update_vertex(v)
        self.compute_shortest_path()

    def update_start(self, new_start):
        if new_start == self.start:
            return
        self.k_m += self.heuristic(self.start, new_start)
        self.start = new_start
        self.compute_shortest_path()