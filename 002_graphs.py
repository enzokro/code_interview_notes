from collections import defaultdict, deque
from types import List

def topological_sort(digraph):
    # digraph is a dictionary:
    #   key: a node
    # value: a set of adjacent neighboring nodes

    # construct a dictionary mapping nodes to their
    # indegrees
    indegrees = {node : 0 for node in digraph}
    for node in digraph:
        for neighbor in digraph[node]:
            indegrees[neighbor] += 1

    # track nodes with no incoming edges
    nodes_with_no_incoming_edges = []
    for node in digraph:
        if indegrees[node] == 0:
            nodes_with_no_incoming_edges.append(node)

    # initially, no nodes in our ordering
    topological_ordering = [] 

    # as long as there are nodes with no incoming edges
    # that can be added to the ordering 
    while len(nodes_with_no_incoming_edges) > 0:

        # add one of those nodes to the ordering
        node = nodes_with_no_incoming_edges.pop()
        topological_ordering.append(node)

        # decrement the indegree of that node's neighbors
        for neighbor in digraph[node]:
            indegrees[neighbor] -= 1
            if indegrees[neighbor] == 0:
                nodes_with_no_incoming_edges.append(neighbor)

    # we've run out of nodes with no incoming edges
    # did we add all the nodes or find a cycle?
    if len(topological_ordering) == len(digraph):
        return topological_ordering  # got them all
    else:
        raise Exception("Graph has a cycle! No topological ordering exists.")



class UnionFind:
    '''Union-Find based Tree data structures. Each element in the set is represented by a node in the tree. The `parent` list stores the root of each Tree, and the `size` list stores the total number of nodes of each tree.

    The `find` operation uses path compression, which makes every node in the tree point to the current root.

    The `union` operation works by rank. That means we make the root of the smaller tree the parent of the larger tree's node. This helps keep the tree balanced.
    '''
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.size[x_root] < self.size[y_root]:
            self.parent[x_root] = y_root
            self.size[y_root] += self.size[x_root]
        else:
            self.parent[y_root] = x_root
            self.size[x_root] += self.size[y_root]


def countComponents(self, n: int, edges: List[List[int]]) -> int:
    '''1D version of counting islands.

    Need one less loop, and two sets of adjacencies because the graph is undirected.
    Also, because of undirected, we need to keep track of the previous node when visiting next.
        Else, we hit a false cycle condition
    '''

    # edge case when we have no nodes or a single one
    if n <= 1: return n
    
    count = 0 # track the number of visited nodes
    visited = set() # keep track of nodes we've visited

    # build the adjacency set
    adj_list = {i: [] for i in range(n)}

    # because the graph is undirected, we need to add both edges to the adjacency list
    for ai,bi in edges:
        adj_list[ai].append(bi)
        adj_list[bi].append(ai) # NOTE: add here since we have undirected nodes, aka it goes both ways
    
    def visit_connected_nodes(node, prev):            
        "Recursively visits nodes reachable from `node`, excluding `prev`."
        
        # base case: we've already visited this node. nothing to do
        if node in visited:
            return
        
        # if it's a new node, then add it to our visited set
        visited.add(node)
        
        # visit its neighbors, keeping track of current `node` to avoid false cycles
        for child in adj_list[node]:
            if child != prev:
                visit_connected_nodes(child, prev=node)

    # visit every node      
    for node in adj_list:
        # if we have not yet visited this node
        if node not in visited:
            # recursively mark all of its neighbors as visited
            visit_connected_nodes(node, -1)
            # increment our count, since finished visiting a new connected component
            count += 1
    
    return count


class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:

        # counts the number of components
        ncomps = 0
        # tracks which nodes have been visited already
        seen = set()

        # add both edges to the adjacency list, so we can visit neighbors
        adj = defaultdict(list)
        for n1,n2 in edges:
            adj[n1].append(n2)
            adj[n2].append(n1)

        def dfs(node):
            '''Depth-first search approach
            
            Start by adding the current node to the visited set
            Then, for each neighbor:
                If we have not seen this node, recrusively visit it
            '''
            seen.add(node)
            for n in adj[node]:
                if n not in seen:
                    dfs(n)

        def bfs(node):
            '''Breadth-first search approach
            
            Start by adding the current node to the queue
            Then, while the q is not empty:
                Pop from the left to make it BFS
                Add this node to the seen set
                Then, for each neighbor:
                    if it has not been seen, put the node on the queue
            '''
            q = deque([node])
            while q:
                n = q.popleft()
                # NOTE: must be here because of undirected graph
                seen.add(n)
                for ni in adj[n]:
                    if ni not in seen:
                        q.append(ni)

        # go through all nodes
        for i in range(n):
            # if we have not visited it, search through
            if i not in seen:
                bfs(i)
                # at this point all connected comps have been connected. Increase the count
                ncomps += 1


        '''Using the Union-Find algorithm above
        '''
        # Initialize the Union-Find data structure
        uf = UnionFind(n)

        # Iterate through the edges in the graph
        for u, v in edges:
            # Combine the subsets containing u and v using the union operation
            uf.union(u, v)

        # Count the number of subsets in the Union-Find data structure
        return len(set(uf.find(i) for i in range(n)))

        return ncomps


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:

        # ocean dimensions
        nrows, ncols = len(grid), len(grid[0])

        # track the set of visited islands
        seen = set()

        def cell_in_bounds(r, c):
            return (0 <= r < nrows) and (0 <= c < ncols)

        def is_new_land(r, c):
            return (r,c) not in seen and grid[r][c] == '1'

        def visit_neighbors(row, col):
            steps = [0, 0, -1, 1]
            for (dr, dc) in zip(steps, steps[::-1]):
                yield row + dr, col + dc

        def bfs(r, c):
            # add current cell to the visited/seen set
            seen.add((r,c))

            # visit cells in iterative BFS
            q = [(r,c)]
            while q:
                
                # pop the most recently added cell
                # NOTE: q.pop(-1) to make it iterative DFS
                row, col = q.pop(0)

                # visit the neighboring horizontal and vertical cells
                for (r, c) in visit_neighbors(row, col):

                    # positive checks: make sure the cell is valid, and that it's unseen land
                    if cell_in_bounds(r, c) and is_new_land(r, c):
                        seen.add((r,c))
                        q.append((r,c))


        # track the total number of islands
        num_islands = 0

        # iterate through all cells in the matrix
        for r in range(nrows):
            for c in range(ncols):
                if is_new_land(r, c):
                    bfs(r, c)
                    num_islands += 1

        return num_islands

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        '''The actual bits of code are straightforward.
        
        2D version of counting connected components, with a fancy name and setup. 
        The heavy lifting is done in BFS, which needs supporting data structures.'''
        # at start, we have no islands
        num_islands = 0
        
        # base case, empty grid with no land or water
        if not grid:
            return num_islands
        
        # setup: rows and columns
        nrows, ncols = len(grid), len(grid[0])
        
        # catch different values of land
        land_vals = (1,'1')
        
        # keep track of nodes we already visited
        visited = set()
        
        # bfs, with dequeue as auxiliary DS
        def bfs(r, c):
            
            # will visit all neighbors of (r,c)
            q = collections.deque()
            
            # mark this node as visited, and put it on the queue to start iterating
            visited.add((r,c))
            q.append((r,c))
            
            # keep going while we have un-visited neighbors
            while q:
                
                # grab current node
                # NOTE: this is where we could change to a DFS, by using pop() instead of popleft()
                #       That way, we visit the most recent node instead of the first.
                row, col = q.popleft()
                
                # check all of its neighbors
                deltas = [0, 0, 1, -1]
                for (dr, dc) in zip(deltas, deltas[::-1]):
                    
                    # neighbor cell to visit
                    r = row + dr
                    c = col + dc 
                    
                    # make sure we are in bounds
                    if (0 <= r < nrows) and (0 <= c < ncols):
                        
                        # if this is land and we have not visited it before, add to queue and mark as visit
                        if grid[r][c] in land_vals and (r,c) not in visited:
                            q.append((r,c))
                            visited.add((r,c))
        
        # iterate through each cell
        for r in range(nrows):
            for c in range(ncols):
                if grid[r][c] in land_vals and (r,c) not in visited:
                    bfs(r, c)
                    # at this point, we've connected (`is in visited`) all nodes touching this one (r,c)
                    num_islands += 1
                      
        return num_islands

    def numIslands(self, grid: List[List[str]]) -> int:
        # dfs recursive with visited set to not modify input
        
        m, n = len(grid), len(grid[0])
        count, visited = 0, set()
        
        # drown the connected landmass 
        def dfs(r, c, grid, visited):
            # this `or` chain could be faster since any condition trips the return
            if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != '1' or (r,c) in visited:
                #     # this recursive keeps going until we reach an invalid cell, or go out of bounds
                # if (not (0 <= r < m)  or  # row out of bounds
                #     not (0 <= c < n)  or  # column out of bounds
                #     grid[r][c] != '1' or  # grid cell is not land
                #     (r,c) in visited      # we've already visited this cell
                # ):
                return
            
            # at this point, this grid cell is an unvisited land within bounds
            # so, add it to visited 
            visited.add((r,c))
            
            # recursively visit this cells neighbors neighbors
            # NOTE: called on each child or (r+/-1, c+/-1) first, aka DFS
            dfs(r + 1, c, grid, visited)
            dfs(r - 1, c, grid, visited)
            dfs(r, c + 1, grid, visited)
            dfs(r, c - 1, grid, visited)

            # # cleaner way of visiting neighbors
            # deltas = [0, 0, 1, -1]
            # for (dr, dc) in zip(deltas, deltas[::-1]):
            #     dfs(r + dr, r + dc, grid, visited)
        
        # iterate over matrix to find an island        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and (i,j) not in visited:
                    count += 1
                    dfs(i, j, grid, visited)                                     
        
        return count




class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        
        # dimensions of matrix
        nrows = len(isConnected)

        # NOTE: we can use a mask since nodes are known in range(n)
        seen = set() # [False] * n
        cnt = 0

        ## with an adjacency list
        # adj_list = defaultdict(list)
        # for r in range(nrows):
        #     for c in range(ncols):
        #         if isConnected[r][c] == 1:
        #             adj_list[r].append(c)
        #             adj_list[c].append(r)

        def bfs(node):
            seen.add(node)
            q = [node]
            while q:
                n = q.pop(0) # pop(-1) to make it iterative dfs
                # for nei in adj_list[n]:
                for nei in [i for i in range(nrows) if i != n and isConnected[n][i]]:
                    if nei not in seen:
                        seen.add(nei)
                        q.append(nei)

        def dfs(node):
            if node not in seen:
                seen.add(node)
            # for nei in adj_list[node]:
            for nei in [i for i in range(nrows) if i != node and isConnected[node][i]]:
                if nei not in seen:
                    dfs(nei)
        
        for node in range(nrows):
        # for node in adj_list:
            if node not in seen:
                dfs(node)
                cnt += 1

        return cnt 


def pacificAtlanticWaterFlow(heights):
    '''Can we reach the ocean?
    Run a DFS from the corners of the array, checking which cells can reach the edges.
    '''
    # dimension of the matrix
    nrows, ncols = len(heights), len(heights[0])
    # sets for visited cells
    pac, atl = set(), set()

    def can_reach_ocean(row, col, visited, prev_height):
        # check if this cell can reach the ocean
        if ((row,col) not in visited and # we haven't seen this yet
            0 <= row < nrows and # the row is within bounds
            0 <= col < ncols and # the col is within bounds
            heights[row][col] >= prev_height): # the current more-inner cell is taller than the prev

            # mark this ocean-reachable-cell as visited
            visited.add((row,col))

            # check all of its neighbors
            deltas = [0, 0, -1, 1]
            for dr,dc in zip(deltas, deltas[::-1]):
                can_reach_ocean(row+dr, col+dc, visited, heights[row][col])


    # check the top/bottom rows can reach their ocean
    for c in range(ncols):
        # visit each column's cells
        can_reach_ocean(0, c, pac, heights[0][c]) # top row
        can_reach_ocean(nrows - 1, c, atl, heights[nrows-1][c]) # bottom row

    # check the left/right cols can reach their ocean
    for r in range(nrows):
        can_reach_ocean(r, 0, pac, heights[r][0]) # left column
        can_reach_ocean(r, ncols-1, atl, heights[r][ncols-1]) # right column

    # return set of cells that can reach both oceans
    return pac & atl # technically to comply with return type: list(map(list, pac & atl))



def courseScheduleOne(numCourses, prerequisites):
    '''Boils down to finding whether a given graph has cycles between courses and pre-reqs'''

    '''First Neetcode solution using a visited hashmap and DFS'''
    # create the visited-set and adjacency list
    visited = set()
    adj_list = {i : [] for i in range(numCourses)}
    # populate the course adjacency list
    for course, pre_req in prerequisites:
        adj_list[course].append(pre_req)
        
        
    def dfs(course):
        
        # found a cycle, cannot complete the course
        if course in visited: 
            return False
        
        # course has no pre-reqs, it can be completed
        # NOTE: using adjacency list as proxy for completed state
        if adj_list[course] == []: 
            return True
        
        # mark this course as visited
        visited.add(course) 
        
        # now, recursively, check the pre-reqs of this current course
        # if this recursively finds `course` in visited again, we hit a cycle and it will return false
        for pre in adj_list[course]:
            if not dfs(pre): return False
            
        # pop it from the visited set, after the recursive call
        visited.remove(course) 
        
        # mark it as done, so we immediately return in following calls, no duplicate work
        # because, if we reached here, we could complete all pre-reqs without a cycle
        adj_list[course] = []
        
        # if we made it here, the course can be completed
        return True
    
    # have to manually check each course, in case the graph is not fully connected
    for course in range(numCourses):
        # couldn't complete a DFS: we found a cycle
        if not dfs(course): return False
        
    return True


    '''Alternative DFS using a state matrix, instead of a visited set, to manually keep track of node states'''
    # populate the course adjacency list
    adj_list = {i : [] for i in range(numCourses)}
    for course, pre_req in prerequisites:
        adj_list[course].append(pre_req)
        
    # state: alternative to visited sets
    state = [0] * numCourses

    def hasCycle(v):
        if state[v] == 1:
            # This vertex is processed so we pass.
            return False
        if state[v] == -1:
            # This vertex is being processed, but we've seen it again, means we have a cycle.
            return True
        # if state[v] in (1,-1): return state[v] == -1

        # Set state to -1, mark it as processing.
        # This is our trip-wire for this node to find cycles when we recursively visit its neighbors
        state[v] = -1

        for i in adj_list[v]:
            if hasCycle(i):
                return True

        state[v] = 1
        return False

    # we traverse each vertex using DFS, if we find a cycle, stop and return
    return not any(hasCycle(v) for v in range(numCourses))
    # return not any(map(hasCycle, range(numCourses)))


    '''Alternative set-solution with DFS, that uses an auxiliary stack to know whether a node has been processed. IIRC, the neetcode solution avoids the stack by setting adjacency to empty list once a node's been processed. The stack is axuiliary to membership in the adjacency list.'''
        # build Adjacency list from Edges list
        adjList = self.buildAdjacencyList(numCourses, prerequisites)
        visited = set()

        def hasCycle(v, stack):
            if v in visited:
                if v in stack:
                    # This vertex is being processed and it means we have a cycle.
                    return True
                # This vertex is processed so we pass
                return False

            # mark this vertex as visited
            visited.add(v)
            # add it to the current stack
            stack.append(v)

            for i in adjList[v]:
                if hasCycle(i, stack): # node being on the stack is alternative to marking its state as -1
                    return True

            # once processed, we pop it out of the stack
            stack.pop()
            return False

        # we traverse each vertex using DFS, if we find a cycle, stop and return
        for v in range(numCourses):
            if hasCycle(v, []):
                return False

        return True


    '''Lastly, find if there is a valid TOPOLOGICAL SORT
    https://www.interviewcake.com/concept/java/topological-sort#:~:text=The%20topological%20sort%20algorithm%20takes,is%20called%20a%20topological%20ordering.
    '''
    def topoBFS(self, numNodes, edgesList):
        # Note: for consistency with other solutions above, we keep building
        # an adjacency list here. We can also merge this step with the next step.
        adjList = self.buildAdjacencyList(numNodes, edgesList)

        # 1. A list stores No. of incoming edges of each vertex
        inDegrees = [0] * numNodes
        for v1, v2 in edgesList:
            # v2v1 form a directed edge
            inDegrees[v1] += 1 # NOTE: in un-directed graphs, we have to link both of them

        # 2. a queue of all vertices with no incoming edge
        # at least one such node must exist in a non-empty acyclic graph
        # vertices in this queue have the same order as the eventual topological
        # sort
        queue = []
        for v in range(numNodes):
            if inDegrees[v] == 0:
                queue.append(v)
        # queue = [v for v in range(numNodes) if inDegrees[v] == 0]
        # queue = [*filter(lambda v: inDegrees[v] == 0, range(numNodes))]

        # initialize count of visited vertices
        count = 0
        # an empty list that will contain the final topological order
        topoOrder = []

        while queue:
            # a. pop a vertex from front of queue
            # depending on the order that vertices are removed from queue,
            # a different solution is created
            v = queue.pop(0)
            # b. append it to topoOrder
            topoOrder.append(v)

            # increase count of visited nodes by 1
            # if we have cycles, we'll visit the same node multiply times and "repeat" its count
            count += 1

            # for each descendant of the current vertex, reduce its in-degree by 1
            for des in adjList[v]:
                inDegrees[des] -= 1
                # if "popping" this node from the graph created a new node with in-degree of 0, add it to queue
                if inDegrees[des] == 0:
                    queue.append(des) # queue += des,

        # if count != numNodes:
        #     return None  # [] #graph has at least one cycle
        # else:
        #     return topoOrder
        # return ([], topoOrder)[count == numNodes]
        return count == numNodes # NOTE: if we need an ordering itself: topoOrder if count == numNodes else []

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # return True if self.topoBFS(numCourses, prerequisites) else False
        # # cleaner check for topo ordering, but need to return empty list when count != numNode
        return self.topoBFS(numCourses, prerequisites)
        

    '''Likely the cleanest DFS we're gonna see, with states'''
    def canFinish(self, N, edges):
        pre = defaultdict(list)
        for x, y in edges: pre[x].append(y)

        status = [0] * N
        def canTake(i):
            if status[i] in {1, -1}: return status[i] == 1
            status[i] = -1
            if any(not canTake(j) for j in pre[i]): return False
            status[i] = 1
            return True

        return all(canTake(i) for i in range(N))


    '''Likely the best commented and intuitive Topological sort we're gonna see'''
    def canFinish(self, numCourses, prerequisites):
        # check empty inputs
        if numCourses <= 0:
            return False

        # build in-degree counter and adjacency lists
        # NOTE: default dict does not work here for some reason...
        inDegree = {i : 0  for i in range(numCourses)}
        adj_list = {i : [] for i in range(numCourses)}

        # intialization with one loop
        # inDegree, adj_list = {}, {}
        # for i in range(numCourses):
        #     inDegree[i], adj_list[i] = 0, []

        # count in-degrees and populate adjacency list
        for child, parent in prerequisites:
            adj_list[parent].append(child)
            inDegree[child] += 1

        # start queue with with nodes of 0 in-degree
        # if no such nodes exist, we immediately have a cycle
        sources = deque(node for node,degree in inDegree.items() if degree == 0)

        # counter for number of nodes we've visited
        visited = 0       
        while sources:
            # grab the current node, and increase number of visitations
            vertex = sources.popleft()
            visited += 1

            # for all descendants, now that we've "poppped" this node from the graph, reduce their indegrees
            for child in graph[vertex]:
                inDegree[child] -= 1
                # if degree reduction creates a new node with in-degree of zero, mark it for processing
                if inDegree[child] == 0:
                    sources.append(child)
        
        # if there was no cycle, we visited all nodes (popped into sources, hence visisted += 1)
        # that means there were no cycles
        return visited == numCourses


    '''An even cleaner and more efficient version of BFS topo'''
    # adjacency and in-degrees, can use this because nodes are integers and can therefore be indexes
    G = [[] for i in range(n)]
    degree = [0] * n
    for j, i in prerequisites:
        G[i].append(j)
        degree[j] += 1
    # use list in place of deque, start with 0-degree nodes
    bfs = [i for i in range(n) if degree[i] == 0]
    # process all nodes
    for i in bfs:
        # process neighbors...
        for j in G[i]:
            # decrement degree
            degree[j] -= 1
            # append to list if we created a 0-degree node, aka completed pre-reqs
            if degree[j] == 0:
                bfs.append(j)
    # if we can complete the classes, we should have visited each node
    return len(bfs) == n


def cloneGraph(self, node: 'Node') -> 'Node':
    '''Main idea is to recursively visit and clone each node, as needed.
    
    Roughly a DFS, but it's very bare bones. Better to call it "clone"
    '''
    
    # stop early if we're given an empty node
    if not node: return None
    
    # map to keep track of which nodes we've cloned
    old2new = {}
    
    def clone_helper(node):
        
        # if we've already cloned this node, return the clone
        if node in old2new:
            return old2new[node]
        
        # else, clone the node and add it to the map
        new = Node(node.val)
        old2new[node] = new
        
        # else, recursively visit and clone the neighbors
        for n in node.neighbors:
            new.neighbors.append(clone_helper(n))
            
        # return the new cloned copy.
        # NOTE: at the top of the recursion stack, this will be the fully cloned graph
        return new
    
    cloned = clone_helper(node)
    return cloned


    '''Slight alternative using the function itself, without a helper, cleaner'''
    def __init__(self): self.old2new = {}
        
    def cloneGraph(self, node: 'Node') -> 'Node':
        '''Main idea is to recursively visit and clone each node, as needed.
        
        Roughly a DFS, but it's very bare bones. Better to call it "clone"
        '''

        # base case 1) no node, return
        if not node: return
        # base case 2) if we already cloned this node, return it
        if node in self.old2new:
            return self.old2new[node]
        
        # create a new node and add its mapping, to mark as "visited"
        new = Node(node.val)
        self.old2new[node] = new
        
        # clone each of this node's neighbors
        # this is where the recursion kicks in and copies all neighbor's ^ n
        new.neighbors = [self.cloneGraph(n) for n in node.neighbors] # list(map(self.cloneGraph, n.neighbors))
        
        # at the end of the recursion stack, this will be the first-clone node, along with all neighbords copied
        return new

    '''Same as above, slightly cleaner code with OOP
    '''
    class Solution(dict):
        def cloneGraph(self, node: 'Node') -> 'Node':
            '''Main idea is to recursively visit and clone each node, as needed.
            
            Roughly a DFS, but it's very bare bones. Better to call it "clone"
            '''
            if not node: return
            
            if self.get(node):
                return self[node]
            
            new = Node(node.val)
            self[node] = new
            
            new.neighbors = [self.cloneGraph(n) for n in node.neighbors]
            
            return new


    '''Same idea but BFS to avoid a potentially large recursion stack.
    Main changes are the auxiliary datastructure, a deque, and popping/adding + cloning/visiting the neighbors'''
    if not node:
        return node

    # Dictionary to save the visited node and it's respective clone
    # as key and value respectively. This helps to avoid cycles.
    visited = {}

    # Put the first node in the queue
    queue = deque([node])
    # Clone the node and put it in the visited dictionary.
    visited[node] = Node(node.val, [])

    # Start BFS traversal
    while queue:
        # Pop a node say "n" from the from the front of the queue.
        n = queue.popleft()
        # Iterate through all the neighbors of the node
        for neighbor in n.neighbors:
            if neighbor not in visited:
                # Clone the neighbor and put in the visited, if not present already
                visited[neighbor] = Node(neighbor.val, [])
                # Add the newly encountered node to the queue.
                queue.append(neighbor)
            # Add the clone of the neighbor to the neighbors of the clone node "n".
            visited[n].neighbors.append(visited[neighbor])

    # Return the clone of the node from visited.
    return visited[node]
            



def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
    '''Similar to Atlantic Water Flow and number of ways of reaching a target cell.

    Except, we can only "climb" cells, so the next value must be larger.
    And, instead of having a target cell to reach, the values around us determine which steps we can take.
    '''
    # Check edge case
    if not matrix:
        return 0

    # Initialize
    rows, cols = len(matrix), len(matrix[0])
    deltas = [0, 0, -1, 1]
    memo = [[None] * cols for _ in range(rows)]
    res = 0

    def dfs(i, j, visited):
        # Check if the result has been cached
        if memo[i][j] is not None:
            return memo[i][j]

        # if we can't move at all, the longest path is itself
        res = 1

        # work with neighbors
        for dr, dc in zip(deltas, deltas[::-1]):
            next_i, next_j = i + dr, j + dc

            # for each direction we try to find a new count
            direction_count = 0
            if (0 <= next_i < rows) and (0 <= next_j < cols):
                # if this new cell is *increasing* compared to current, aka we can take the step
                if matrix[next_i][next_j] > matrix[i][j]:
                    # add one to direction count and visit its neighbors
                    direction_count = 1 + dfs(next_i, next_j, visited)

            # update locally with max result from following the paths
            res = max(direction_count, res)

        # cache the result, and return it
        memo[i][j] = res
        return res

    # visit each cell
    for row in range(rows):
        for col in range(cols):
            # find the maximum so far
            res = max(dfs(row, col, memo), res)

    return res



def exist(self, board: List[List[str]], word: str) -> bool:
    '''Backtracking solution, with brutal Time Complexity.
    Gives TLE exceeded on leetcode
    
    Brutal complexity: 
        O(n * m * 4^^len(words))
    '''
    
    path = set()
    deltas = [0, 0, -1, 1]
    nrows, ncols = len(board), len(board[0])
    
    def search_word(r, c, idx, word):
        
        # found the word!
        if len(word) == 0:
            return True
            
        # base cases: (r,c) out of bounds, or the character is wrong, or we're looking at the same character
        if ((r,c) in path or 
            not(0 <= r < nrows) or 
            not(0 <= c < ncols) or
            board[r][c] != word[0]):
                return False
        
        # add path to hash set to avoid re-visiting the previous cell
        path.add((r,c))
        
        res = [search_word(r+dr, c+dc, idx + 1, word[1:]) for dr,dc in zip(deltas,deltas[::-1])]
        # for dr, dc in zip(deltas, deltas[::-1]):
        #     res.append(search_word(r+dr, c+dc, idx + 1, word[1:]))
        
        # done processing, can pop off the visited stack
        path.remove((r,c))
        
        return any(res)
        
        # for r in range(nrows):
        #     for c in range(ncols):
        #         # search every cell for the first letter
        #         if search_word(r, c, 0, word):
        #             return True
        # return False
        # one-liner
        return any(search_word(r, c, 0, word) for r in range(nrows) for c in range(ncols))

        '''Below an excellent solution that actually runs.
        Seems it uses Counter for computation savings'''
        R = len(board)
        C = len(board[0])
        
        # if len of word is greater than total number of character in board
        if len(word) > R*C:
            return False
        
        count = Counter(sum(board, []))
        
        # count of a LETTER in word is Greater than count of it being in board
        for c, countWord in Counter(word).items():
            if count[c] < countWord:
                return False
            
        # if count of 1st letter of Word(A) is Greater than that of Last One in Board(B). 
        # Reverse Search the word then search as less case will be searched.
        if count[word[0]] > count[word[-1]]:
             word = word[::-1]
                        
        # simple backtracking 
        seen = set()    # so we dont access the element again
        
        def dfs(r, c, i):
            if i == len(word):
                return True
            if r < 0 or c < 0 or r >= R or c >= C or word[i] != board[r][c] or (r,c) in seen:
                return False
            
            seen.add((r,c))
            res = (
                dfs(r+1,c,i+1) or 
                dfs(r-1,c,i+1) or
                dfs(r,c+1,i+1) or
                dfs(r,c-1,i+1) 
            )
            seen.remove((r,c))  #backtracking

            return res
        
        for i in range(R):
            for j in range(C):
                if dfs(i,j,0):
                    return True
        return False
    
        '''With some tiny final edits for mayyyybe speedups? In Python 11 perchance'''
        R = len(board)
        C = len(board[0])
        deltas = [0, 0, -1, 1]
        
        # if len of word is greater than total number of character in board
        if len(word) > R*C:
            return False
        
        count = Counter(sum(board, []))
        
        # count of a LETTER in word is Greater than count of it being in board
        for c, countWord in Counter(word).items():
            if count[c] < countWord:
                return False
            
        # if count of 1st letter of Word(A) is Greater than that of Last One in Board(B). 
        # Reverse Search the word then search as less case will be searched.
        if count[word[0]] > count[word[-1]]:
             word = word[::-1]
                        
        # simple backtracking 
        seen = set()    # so we dont access the element again
        
        def dfs(r, c, i):
            if i == len(word):
                return True
            if r < 0 or c < 0 or r >= R or c >= C or word[i] != board[r][c] or (r,c) in seen:
                return False
            
            seen.add((r,c))
            res = [dfs(r+dr, c+dc, i + 1) for dr,dc in zip(deltas,deltas[::-1])]
            seen.remove((r,c))  #backtracking

            return any(res)
        
        return any(dfs(r, c, 0) for r in range(R) for c in range(C))




def alienDictionary(words):
    '''
    We are given a list of strings that follow a lexicographical order.
        Aka, the way strings are usually sorted a-z
    But, this is actually an alien language we don't recognize!
    We need to return the characters in proper lexicographical order.

    There are cases that make it impossible:
        Repeated words, i.e. [a , c, a] <- impossible if sorted
        When the sorting is wrong: [abcde, abc] <- impossible, abc should be first

    The alien sorting gives us a way to build a graph.
    The main idea:
        - Build an adjacency list from the characters and words.
        - Check if the graph has a topological sorting
            - Can be BFS/DFS
        
    From definition of topological sorting, the nodes will be "in order" given the sorting.

    If we ever find a cycle, we can immediately return.

    Question: why do we need to paint graph below, why doesn't set membership work?
    '''
    # build the adjacency set
    adj_list = {c: set() for word in words for c in word}

    # process word-pairs to find letter sortings
    for w1,w2 in zip(words,words[1:]):

        # check if sorting is invalid (w1 is longer but their prefixes are the same)
        minLen = min(len(w1), len(w2))

        # NOTE: have to check for invalid sorting here, based on lens and prefixes
        if len(w1) > len(w2) and w1[:minLen] == w2[:minLen]:
            return ''

        # build the character adjacency graph
        for c1,c2 in zip(w1,w2):
            if c1 != c2:
                # find the first non-equal character and mark its sorting
                adj_list[c1].add(c2)
                break # the other, remaxining suffixes don't matter

    # now, we are ready to check if this adjacency list has a topological sorting
    visited = {} # to check for cycles with graph painting, two colors: (processing, done)

    # final response
    res = []

    def has_cycle(char):

        # if we've seen this character before, check if it's from a looped-cycle or new, separate loop
        if char in visited:
            return visited[char]

        # mark this character as processing (paint it grey)
        visited[char] = True

        # check all neighbors
        for node in adj_list[char]:
            if has_cycle(node):
                return True

        # we are done (paint it black)
        # NOTE: would noramlly return False here, but we keep the state to allow post-order appending below
        visited[char] = False

        # post-order add to result
        res.append(char)
        # implicit return of None as False (for `if` check) 


    # check all of the nodes in case we don't have a fully connected graph
    # not purely functional: `has_cycle` has side-effect of post-order DFS insertion
    if any(map(has_cycle, adj_list)):
        return ''

    # return the reversed order string
    return ''.join(res[::-1])
