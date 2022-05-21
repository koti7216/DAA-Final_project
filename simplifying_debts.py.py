#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install multipledispatch


# In[46]:


from collections import deque 
from typing import List
from multipledispatch import dispatch

import sys

class Main :
    OFFSET = 1000000000
    visitedEdges = None
    @staticmethod
    def main(args) :
        Main.Graph_Debts()
    
    # Here Alice, Bob, Charlie, David, Ema, Fred and Gabe are represented by vertices from 0 to 6 respectively.
    @staticmethod
    def Graph_Debts() :
        person = ["Alice", "Bob", "Charlie", "David", "Ema", "Fred", "Gabe"]
        n = len(person)
        solver = Dinic(n, person)
        solver = Main.addAllTransactions(solver)
        print()
        print("Simplify Debts...")
        print("--------------------")
        print()
        Main.visitedEdges =  set()
        edgePos = None
        #edgePos = Main.getNonVisitedEdge(solver.getEdges())
        while (Main.getNonVisitedEdge(solver.get_edges()) != None) :
            edgePos = Main.getNonVisitedEdge(solver.get_edges())
            solver.reevaluate()
            firstEdge = solver.get_edges()[edgePos]
            solver.set_Source(firstEdge.start)
            solver.set_Sink(firstEdge.to)
            residualGraph = solver.get_graph()
            newEdges =  []
            for allEdges in residualGraph :
                for edge in allEdges :
                    remainingFlow = (edge.capacity if (edge.flow < 0) else (edge.capacity - edge.flow))
                    if (remainingFlow > 0) :
                        newEdges.append(Dinic.Edges(edge.start, edge.to, remainingFlow))
            maxFlow = solver.getMaxFlow()
            source = solver.Source()
            sink = solver.Sink()
            Main.visitedEdges.add(Main.getHashKeyForEdge(source, sink))
            solver = Dinic(n, person)
            solver.add_edges(newEdges)
            solver.add_edge(source, sink, maxFlow)
        solver.printEdges()
        print()
    @staticmethod
    def  addAllTransactions( solver) :
        solver.add_edge(1, 2, 40)
        solver.add_edge(2, 3, 20)
        solver.add_edge(3, 4, 50)
        solver.add_edge(5, 1, 10)
        solver.add_edge(5, 2, 30)
        solver.add_edge(5, 3, 10)
        solver.add_edge(5, 4, 10)
        solver.add_edge(6, 1, 30)
        solver.add_edge(6, 3, 10)
        return solver

    @staticmethod
    def  getNonVisitedEdge( edges) :
        edgePos = None
        curEdge = 0
        for edge in edges :
            if (not Main.getHashKeyForEdge(edge.start, edge.to) in Main.visitedEdges) :
                edgePos = curEdge
            curEdge += 1
        return edgePos

    @staticmethod
    def  getHashKeyForEdge( u,  v) :
        return u * Main.OFFSET + v


# In[47]:


class Network_Flow :
    INF = sys.maxsize / 2
    class Edges :
        start = 0
        to = 0
        startLabel = None
        toLabel = None
        residual = None
        flow = 0
        cost = 0
        capacity = 0
        originalCost = 0
        def __init__(self, start,  to,  capacity, cost=0) :
            self.start = start
            self.to = to
            self.capacity = capacity
            self.originalCost =self.cost = cost
       # def __init__(self, start,  to,  capacity) :
        #    self.this(start, to, capacity, 0)
        def  isResidual(self) :
            return self.capacity == 0
        def  remainingCapacity(self) :
            return self.capacity - self.flow
        def augment(self, bottleNeck) :
            self.flow += bottleNeck
            self.residual.flow -= bottleNeck
        def  toString(self, s,  t) :
            u = "s" if (self.start == s) else ("t" if (self.start == t) else "".join(self.start))
            v = "s" if (self.to == s) else ("t" if (self.to == t) else "".join(self.to))
            return String.format("Edge %s -> %s | flow = %d | capacity = %d | is residual: %s",u,v,self.flow,self.capacity,self.isResidual())
    n = 0
    s = 0
    t = 0
    maxFlow = 0
    minCost = 0
    minCut = None
    graph = None
    vertexLabels = None
    edges = None
    visitedToken = 1
    visited = None

    solved = False
    def __init__(self, n,  vertexLabels) :
        self.n = n
        self.initial_graph()
        self.assign_labels(vertexLabels)
        self.minCut = [False] * (n)
        self.visited = [0] * (n)
        self.edges =  []
        
    def initial_graph(self) :
        self.graph = [[]] * (self.n)
        
    def assign_labels(self, vertexLabels) :
        if (len(vertexLabels) != self.n) :
            raise ValueError("you must pass %s of values", (self.n))
        self.vertexLabels = vertexLabels

    def add_edges(self, edges) :
        if (edges == None) :
            raise ValueError("Edge can not be null")
        for edge in edges :
            self.add_edge(edge.start, edge.to, edge.capacity)
    
    @dispatch(int,int,int)
    def add_edge(self, start,  to,  capacity) :
        if (capacity < 0) :
            raise ValueError("capacity > 0")
        e1 = Network_Flow.Edges(start, to, capacity)
        e2 = Network_Flow.Edges(to, start, 0)
        e1.residual = e2
        e2.residual = e1
        self.graph[start].append(e1)
        self.graph[to].append(e2)
        self.edges.append(e1)        
            
    @dispatch(int,int,int,int)
    def add_edge(self, start,  to,  capacity,  cost) :
        e1 = NetworkFlowSolverBase.Edge(start, to, capacity, cost)
        e2 = NetworkFlowSolverBase.Edge(to, start, 0, -cost)
        e1.residual = e2
        e2.residual = e1
        self.graph[start].append(e1)
        self.graph[to].append(e2)
        self.edges.append(e1)
    
    def visit(self, i) :
        self.visited[i] = self.visitedToken

    def  visited(self, i) :
        return self.visited[i] == self.visitedToken

    def Nodes_unvisited(self) :
        self.visitedToken += 1

    def  get_graph(self) :
        self.exec_()
        return self.graph

    def  get_edges(self) :
        return self.edges
    # Returns the maximum flow from the source to the sink.
    def  getMaxFlow(self) :
        self.exec_()
        return self.maxFlow
    # Returns the min cost from the source to the sink.
    # NOTE: This method only applies to min-cost max-flow algorithms.
    def  MinCost(self) :
        self.exec_()
        return self.minCost

    def  MinCut(self) :
        self.exec_()
        return self.minCut

    def set_Source(self, s) :
        self.s = s

    def set_Sink(self, t) :
        self.t = t

    def  Source(self) :
        return self.s

    def  Sink(self) :
        return self.t

    def reevaluate(self) :
        self.solved = False

    def printEdges(self) :
        for edge in self.edges :
            print(str.format("%s ----%s----> %s",self.vertexLabels[edge.start],edge.capacity,self.vertexLabels[edge.to]))

    def exec_(self) :
        if (self.solved) :
            return
        self.solved = True
        self.solver()

    def solver(self) :
        pass


# In[48]:


class Dinic(Network_Flow) :
    level = None

    def __init__(self, n,  vertexLabels) :
        super().__init__(n, vertexLabels)
        self.level = [0] * (n)
    def solver(self) :
        next = [0] * (self.n)
        while (self.bfs()) :
            next = [0]*len(next)
            f = self.dfs(self.s, next, Network_Flow.INF)
            while (f != 0) :
                self.maxFlow += f
                f = self.dfs(self.s, next, Network_Flow.INF)
        i = 0
        while (i < self.n) :
            if (self.level[i] != -1) :
                self.minCut[i] = True
            i += 1

    def  bfs(self) :
        self.level=[-1]*len(self.level)
        self.level[self.s] = 0
        q =  deque(maxlen=self.n)
        q.append(self.s)
        while (not len(q)) :
            node = q.poll()
            for edge in self.graph[node] :
                cap = edge.remainingCapacity()
                if (cap > 0 and self.level[edge.to] == -1) :
                    self.level[edge.to] = self.level[node] + 1
                    q.append(edge.to)
        return self.level[self.t] != -1
    def  dfs(self, at,  next,  flow) :
        if (at == self.t) :
            return flow
        numEdges = self.graph[at].size()
        while (next[at] < numEdges) :
            edge = self.graph[at].get(next[at])
            cap = edge.remainingCapacity()
            if (cap > 0 and self.level[edge.to] == self.level[at] + 1) :
                bottleNeck = self.dfs(edge.to, next, min(flow, cap))
                if (bottleNeck > 0) :
                    edge.augment(bottleNeck)
                    return bottleNeck
            next[at] += 1
        return 0

    

if __name__=="__main__":
    Main.main([])


# In[ ]:




