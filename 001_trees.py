r'''
# Tree Patterns

These problems deal with traversing a Tree data structure.
There are two main ways to traverse a Tree:
    - Depth-first search
    - Breadth-first search

    
## Depth-First Search

In depth-first traversal the goal is to first go as deep as possible.
Depth-first follows a recursive pattern that we can leverage:
    - If the current node exists (is not None):
        - We process the current node.
        - Then we process its left and/or right children
    - When a node does not exist, we return() (we've hit the recursion's stopping criteria)

The order in which the node and its children are processed defines the kind of depth-first traversal. There are three main types of depth-first traversals:
    - "Pre-Order":   node      -> node.left  -> node.right
    - "In-Order":    node.left -> node       -> node.right
    - "Post-Order":  node.left -> node.right -> node


## Breadth-First Search 

The goal in a breadth-first search is to clear out a "level" in the tree before moving on to the next one. Here is an example of two nodes on the same level in a Tree:
             R
            / \
       ----A---B--- # <-- Nodes A and B are on the same level 

A level is defined as all the nodes that fall on the same horizontal line drawn from any of its nodes. 


Breadth-first search is usually implemented in an iterative (as opposed to recursive) way. For this we need a helper data structure (usually a list or queue) that stores the children on the current level.


## Approaching Tree Problems

We will have some conditions or restraints: i.e. certain values that nodes must have, or their position in the tree, etc. These constraints will force us to modify our iterative solution to solve the problem. In essence, problems are testing for:
    - Whether we know the basic pattern
    - If we can build on the pattern to solve a more challenging problem.
    
    
## Summary and Notes

Tree problems can be popular in interviews which makes them a good starting point. But there are other benefits as well. 

Trees are technically a subset of Graph problems. And Graphs are arguably the most useful topic to know. Thankfully, tree problems are a great introduction to them.

A Tree is a Graph with certain conditions on:
    - The number of incoming edges to a node.
    - The number of outgoing edges from a node.
    - The connectedness of the overall graph. 

More broadly, there seems to be a powerful natural progression from learning/tackling problem topics in this order:
    Trees -> Graphs -> Back-tracking -> Dynamic Programming  

These areas build on each other and can help you see broader patterns while improving your coding problem-solving skills.
 
If you know the four problem topics above, and have a working knowledge of data structures (arrays, linked-lists, queues, heaps), you are ~70% ready for just about any junior+ coding interview. 

Some other problem areas to bring up that readiness to ~90%:
    - Binary Search ** # <-- great problem area, many sharp edges that lead to helpful tools
    - Two-Pointer problems 
    - Sliding Window problems


And lastly, the more niche areas, which *might* be good to know. But you should not heavily prioritize (and definitely not procrastinate) on them:
    - Interval problems
    - Cyclic Sort problems
    - Matrix traversals (bonus: most non-traversal Matrix problems are already covered by Graphs!)


## General tips for Problem Areas    

Most of the topic areas can be broken down as follows:

- There are one or more template solutions for each problem topic.
- A template can have a simple version (for Easy problems) and complex versions (for Medium+). 


- The Easy problems mostly implement the simple templates. But many times, we need one or more changes to the template to work under a new constraint given by the problem.  

- Often, this change involves doing some extra work like:
- - Checking for a new condition and making a decision about it.
- - Modifying a loop. Either the looping condition or the code inside it (or both)
- - Sorting the data first.  
- - Bringing in auxiliary data structures (usually hashmaps or sets) to remove unnecessary or expensive work.  

The Medium templates can combine one or more changes from the Easy problems to start. Then, more constraints are added. As more or too-strict constraints are added, the problem grows from Easy -> Medium -> Hard. 

Often the jump from Medium -> Hard problems involves finding key problem-solving insights, possibly given by the limitations of the input and/or the data structures.  

### Final thoughts on incremental LeetCode problems  

There is a type of problem series, where the next problem is always harder than the last one. For example, there is a problem series "House Robber" with Easy -> Medium -> Hard problems roughly shown as:
- House Robber 1 -> Easy
- House Robber 2 -> Easy/Medium
- House Robert 3 -> Medium/Hard
- House Robber 4 -> Hard  

This is a soft mapping. The difficulty jumps in different problem series are not the same. Many problem series quickly build up to a Hard template. 


Once we are dealing with a Medium/Hard problem template, there is often a nice bonus.

## Conclusion  

Quick recap of what we saw:

1) Focus on understanding the templates for the different problem topics.
2) Practice applying the templates. This part is key

** NOTE: You could ready many books about riding a bike. But until you hop on one, you won't really know *how* to ride a bike. Only how one rides, in theory. At some point you have to get on that bike and pedal(?). The sooner the better.

3) Get comfortable with "Easy" problem constraints, even 2+ at a time.
4) Get comfortable modifying the template for Medium problems
5) Practice problem-solving skills/insights for Medium -> Hard problems

Lastly, make sure to focus on the larger problem-solving patterns across the different topic areas. There is a "formula" to all of this. Which means there is a lot of useful leverage across topic areas. 

Doing all of the above will most importantly help you prepare for a live test, applying a kind of constraint (or many) that you've never seen before. Knowing how to go from Steps 1 -> 5 and back will help you handle these novel constraints. 
'''

# helper libraries
from collections import defaultdict, deque
from typing import List, Optional


# Typical definition for a binary tree node you will see in problems.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val # value of the node
        self.left = left # the node's left child
        self.right = right # the node's right child


'''Recursive pre-order traversal'''
def pre_order_traversal(root: TreeNode):
    '''Root, then left, then right
    '''
    if root:
        print(root.val) # print this value
        in_order_traversal(root.left) # traverse the left
        in_order_traversal(root.right) # traverse the right
    # implicit return when node is none


'''Recursive in-order traversal'''
def in_order_traversal(root: TreeNode):
    '''Left, then root, then right.
    '''
    if root:
        in_order_traversal(root.left) # traverse the left
        print(root.val) # print this value
        in_order_traversal(root.right) # traverse the right
    # implicit return when node is none


'''Recursive post-order traversal'''
def post_order_traversal(root: TreeNode):
    '''Left, then right, then root.
    '''
    if root:
        in_order_traversal(root.left) # traverse the left
        in_order_traversal(root.right) # traverse the right
        print(root.val) # print this value
    # implicit return when node is none




class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        
        def left_leaf_helper(is_left, node):
            # start with a sum of zero for this node
            val = 0

            # base case: empty node, return 0
            if not node:
                return val
            
            # if this node is in a left-branch
            if is_left:
                # if this left-node does not have any children (aka it's a leaf node)
                if not(node.left or node.right):
                    return node.val
            # # could consolidate this statement
            # if is_left and not(node.left or node.right):
            #     return node.val
                
            # process the children
            if node.left: # count up the sum of left's left-leaves
                val += left_leaf_helper(True, node.left)
            if node.right: # sum up the right's left-leaves
                val += left_leaf_helper(False, node.right)

            # return the sum of this node's left-leaves
            return val
                
        # find the sum of all left leaves
        left_sum = left_leaf_helper(False, root)
        return left_sum



class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        def is_bst(node, left_val=float('-inf'), right_val=float('inf')):

            if not node:
                return True

            if node.val <= left_val or node.val >= right_val:
                return False

            left_valid = is_bst(node.left, left_val, min(node.val, right_val))
            right_valid = is_bst(node.right, max(left_val, node.val), right_val)

            return left_valid and right_valid

        return is_bst(root)


        '''Leveraging idea that in-order traversal of BST gets you a monotonically increasing sequence.
        '''
        if not root:
            return True
        def in_order(node):
            if node:
                # if the left sub-tree is not in-order (monotonically increasing), return False
                if not in_order(node.left):
                    return False

                # if the current node's value is smaller than the previous one (another base-case), return False
                if node.val <= self.prev:
                    return False
                # prepare the next previous value
                self.prev = node.val

                # work down the right sub-tree
                return in_order(node.right)

            # base-case: empty node is true
            else:
                return True
        
        # sequence must be monotonically increasing: start from negative value
        self.prev = float('-inf')
        return in_order(root)




class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        '''A valid tree:

            - Has no cycles (topological sort)
            - Is fully connected
                - aka starting from any node, we could have visited all the rest.

        Both conditions can be checked via topological sort
        '''
        if not n: return True # technically, an empty tree is valid (base case?)
        
        # build the adjacency set
        adj_list = {i: [] for i in range(n)}

        # build both sets of adjacencies since graph is undirected
        for ai,bi in edges:
            adj_list[ai].append(bi)
            adj_list[bi].append(ai) # NOTE: add here since we have undirected nodes, aka it goes both ways

        # if the graph is a valid tree, we will have visited all nodes [0,...,n-1]
        visited = set()

        def has_cycle(node, prev=-1):
            "Standard DFS checking for a cycle"

            # if we are re-visiting a node, that means we found a cycle.
            if node in visited:
                return True

            # mark this character as processing (paint it grey)
            visited.add(node)

            # check all neighbors
            for child in adj_list[node]:
                # only check new nodes for cycles, don't revisit the node we just came from
                if child != prev and has_cycle(child, prev=node):
                    return True
            
            # if we made it here, there are no cycles
            return False
        
        # NOTE: the magic happens here. Is this is a valid tree, there are no cycles and it is fully connected.
        # That means we should be able to reach any node from any other node
        # And, because we are using a set, the duplicate additions don't matter. 
        # return the reversed order string
        return (not has_cycle(node=0, prev=-1)) and (len(visited) == n)
            



class Solution:
    '''Classic: invert the binary tree
    As always: base-cases

    If we've reached a null node, nothing to invert
    Else, we have a non-null node and we need to inver its left and right sub-trees
    Working down the recursion stack, we eventually hit leaf nodes with no children,
        Then work our way up to invert the leaf's parent,
        Then recursively back up to invert all the other values

    Note: we are not swapping points, just moving values about
    '''
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        
        def invert_nodes(root):
            "HINT: think of the perspective of a single node! Edge cases"
            # if this node is non-null, we need to invert its children
            if root:
                # swap this node's children
                # worst-case: use a tmp for left
                root.right, root.left = root.left, root.right
                # invert the corresponding sub-trees
                invert_nodes(root.left)
                invert_nodes(root.right)

            return # implicit return (base case) when we hit a null node

        # invert the tree "in place" and return it
        invert_nodes(root)
        return root


class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        '''There is an intuitive solution below where we iterate the entire tree.

        But, we can save some work by using the BST property of the input.
        I.E. given a common ancestor, then unless we are at the LCA, they will both be on one side of tree.
            And! Once they are not, we know that we are at the LCA.

        Improvement:
            Recursively check if nodes are in the left or right subtree.
            If they are not, that means we have reached the branch/split and found the LCA.
        '''
        # base case: empty tree
        if not root: return None
        
        # map from nodes 2 their parents
        node2parent = {}
        
        def get_parents(node, prev):
            if node:
                # A node's parents are it's previous, plus itself
                # Add itself to the end since this would be its more recent ancestor.
                node2parent[node.val] = node2parent.get(prev, []) + [node]
                # recursively get the other parents
                # NOTE: this work is repeated, i.e we're potentially finding parents for an entire root-subtree we don't need to
                get_parents(node.left, node.val)
                get_parents(node.right, node.val)
            
        # find parents for all nodes (O(n) traversal)
        get_parents(root, None)

        # get parents of both
        q_parents = node2parent[q.val]
        p_parents = node2parent[p.val]
        ind = min(len(q_parents), len(p_parents))
        
        # find the matching node, starting from the back, if it exists
        for i in range(ind - 1, -1 , -1):
            if q_parents[i].val == p_parents[i].val:
                break

        return q_parents[i]

        '''Avoid duplicate work
        Recursively check if we need to go left/right subtree, until we don't
            That means we found the split, which will be the LCA

        NOTE: leverage BST properties, and membership reqs
        '''
        # base case: empty tree
        if not root: return None

        # map from nodes 2 their parents
        node2parent = {}

        def find_lca(node, p, q):
            if node:
                # get the values for comparison
                parent_val = node.val
                pval = p.val
                qval = q.val
                
                # check if we need to go right
                if parent_val < pval and parent_val < qval:
                    return find_lca(node.right, p, q)
                # check if we need to go left
                elif parent_val > pval and parent_val > qval:
                    return find_lca(node.left, p, q)
                # else, the nodes are in different subtrees. working from the top-down, this will be the LCA
                else:
                    return node
            # implicit return None when we hit the end

        # find parents for all nodes (O(n) traversal)
        return find_lca(root, p, q)


        '''Iterative, traversing the tree until we find the node'''
        # base case: empty tree
        if not root: return None

        # for convenience grab the two pointers
        p_val, q_val = p.val, q.val
        
        # iterate through the tree until we hit the condition
        node = root
        
        while node:
            
            # have to go left, both values are smaller
            if p_val < node.val and q_val < node.val:
                node = node.left
                
            # have to go right, both values are larger
            elif p_val > node.val and q_val > node.val:
                node = node.right
                
            # else, the nodes are on opposite subtrees, by definition this is the LCA
            else:
                return node
            
        # if we never found it, implicit return of none
            

'''Checking if a binary search tree is height balanced.
That means that the height of subtrees are less than 1 apart, and that all of the subtrees are balanced as well

Need two pieces:
    Finding height of a tree
    Base cases and conditions for being balanced
'''
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        
        # base case: empty tree
        if not root: return True
        
        def height(node):
            return 1 + max(height(node.left), height(node.right)) if node else -1
        
        def is_balanced(node):
            '''Think through edge cases'''
            if not node:
                return True
            
            # # an empty tree is balanced
            # # if the heights differ by more than one, they are not
            # if abs(height(node.left) - height(node.right)) <= 1 and is_balanced(node.left) and is_balanced(node.right):
            #     return True
            # else:
            #     return False

            # simplified one-liner 
            # NOTE: have a lot of work, finding height of sub-trees at each run
            return abs(height(node.left) - height(node.right)) <= 1 and is_balanced(node.left) and is_balanced(node.right)
        
        return is_balanced(root)


        '''Alternative combining height and balance in one call'''
        def dfs(node):
            '''
            return the number of levels below node, counting node
            as 1.  Returns 0 if no node.
            '''
            if not node:
                return 0
            left_height = dfs(node.left)
            right_height = dfs(node.right)
            if abs(right_height - left_height) > 1:
                raise ValueError(f'Unbalanced at {node}')
            return 1 + max(left_height, right_height)
        
        try:
            dfs(root)
            return True
        except ValueError:
            return False


class Solution:
    '''Finding max depth of a binary search tree'''
    def maxDepth(self, root: Optional[TreeNode]) -> int:

        # recursive DFS
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right)) if root else 0
        
    
        ## BFS 2) Without passing lvl or taking max
        if not root: return 0
        # start at the floor level, and put the root on the queue
        level = 0
        q = deque([root])
        
        # while there are nodes to process
        while q:
            
            # clean out this entire level, and add non-children
            for i in range(len(q)): # take a snapshot per-level, regardless of children added
                node = q.popleft()
                # note: None nodes are never added
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
                    
            # after we've cleared a level, we can increment our height
            level += 1
        
        return level
    
    
        ## 3) Iterate Pre-order DFS
        '''Need to use a stack to emulate the recursion stack call! This is the key insight'''

        # BFS A) tracking level in the queue       
        # base case: empty node
        if not root: return 0
        # NOTE: can remove this, by setting `h` to 0 first. Then, the stack runs once, but the update statement never executes because the node is null
        # h = 0
        
        # setup variables
        stack = [(root,1)] # because we are starting with pre-order, can start the queue with the root node
        h = 1 # stores the maximum height seen so far. we at least have the root
        
        while stack:
            # pop the current element and its level
            node, lvl = stack.pop()
        
            # if the node is non-null run the operations
            if node:
                
               # is this node on a new, higher level?
                h = max(lvl, h)
                
                # add its non-null children
                # NOTE: can technically add them regardless, and if node above keeps things valid
                if node.left:
                    stack.append((node.left, lvl + 1))
                if node.right: 
                    stack.append((node.right, lvl + 1))
        
        return h
    

        '''Cleaned up Iterative pre-order DFS'''
        h = 0 # start from the ground floor
        # setup variables
        stack = [(root,1)] # because we are starting with pre-order, can start the queue with the root node
        
        while stack:
            # pop the current element and its level
            node, lvl = stack.pop()
        
            # if the node is non-null run the operations
            if node:
                # is this node on a new, higher level?
                h = max(lvl, h)
                
                # add its children
                stack.append((node.left,  lvl + 1))
                stack.append((node.right, lvl + 1))
        
        return h



class Solution:
    '''DFS recursive search checking if two nodes are the same.

    NOTE: base cases are king! Check:
        If both nodes are non-null
        If only one is non-null
        If their values are not the same

    If none of the base cases hit, we have two non-null nodes with the same values
        Again, the pattern of excluding or checking what we *dont* want.

    Now we need both sub-trees to be the same!
    '''
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        
        def dfs(a, b):
            
            ## Base cases:
            # both nodes are null, technically the same tree
            if a is None and b is None:
                return True
            # only one node is none, they can't be the same
            if (bool(a) != bool(b)): #or (a.val != b.val):
                return False
            # they are both non-null, but their values are not the same
            if a.val != b.val:
                return False
            
            # at this point, nodes are both non-null, need to check their subtrees
            # don't need to worry about their null-ness, we check for all cases in the base 
            return dfs(a.left, b.left) and dfs(a.right, b.right)
        
        return dfs(p, q)
        

'''Template for DFS search'''
def dfs(root, target):
    # base case: root is none, return
    if root is None: return None

    # other base case: if we found the target, return it
    if root.val == target: return root

    # if we did not find it at the root, the target could be in the left subtree
    left = dfs(root.left, target)
    # if we found something return it
    if left is not None:
        return left

    # by now, we know it is not in the left, but it still could be in the right
    # we can return the right's call directly, because if we would have found it by now, we'd have returned earlier
    # if it finds the target, it will have the answer.
    # if it does not find the target, then target !exists in root, left, right, aka it's not here
    return dfs(root.right, target)

    # # NOTE: these last two pieces can be:
    # return dfs(root.left, target) or dfs(root.right, target)
    # if the target is not found, these will be a chain of nested `None or None`
    # but, if the target is found, one of those values will be true, so it will "bubble" up from the `or`.



class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:

        if not root:
            return root

        # stores nodes iterated by column
        nodes = defaultdict(list)

        # initial level and queue with the root node
        lvl = 0
        q = deque([(root,lvl)])

        # while we have levels to traverse
        while q:
            # grab the current node
            node, l = q.popleft()

            # if this node exists
            if node:
                # append its value
                nodes[l].append(node.val)

                # put its children on the queue
                if node.left:
                    q.append((node.left, l-1))
                if node.right:
                    q.append((node.right, l+1))

        # # this breaks the top-down, left-right convention
        # # due to post/pre/in-order, one of the branches will always miss the criteria
        # def get_col_level(node, lvl):
        #     if node:
        #         nodes[lvl].append(node.val)
        #         get_col_level(node.left, lvl - 1)
        #         get_col_level(node.right, lvl + 1)
        #     else:
        #         return
        # get_col_level(root, 0)
        
        return [nodes[lvl] for lvl in sorted(nodes)]