class Solution:
    '''Reversing a linked list:

    Details are as usual in the pointer math
    **Takeaway:: Make sure the order/update is correct, can't be willy nilly.
    Have to make sure head and new links are sorted before moving on to "next" elem
    '''
    def reverseList(self, head: ListNode) -> ListNode:
        last = None
        while head:
            # keep the next node
            tmp = head.next
            # reverse the link
            head.next = last
            # update the last node and the current node
            last = head
            head = tmp
        
        return last


'''Finding cycle in a linked list.
Have two approaches, can use a hash set to check for membership
Or, using fast/slow pointers that will always meet
'''
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        
        # # naive approach using a hashet
        # seen = set()
        # node = head
        # while node:
        #     if node in seen:
        #         return True
        #     seen.add(node)
        #     node = node.next
        # return False
    
        # using fast/slow pointer
        fast = slow = head
        while fast and fast.next:
            # move up the pointers
            fast = fast.next.next # + 2
            slow = slow.next # + 1
            # NOTE: gap always closes by (2 - 1 = 1) -> can decrement from an N + M to N in M steps. They will always meet.
            if fast is slow:
                return True
        return False


class Solution:
    def reverseBetween(self, head: Optional[ListNode], left_pos: int, right_pos: int) -> Optional[ListNode]:
        '''NOTE: I had the right idea but was missing a few details:

        The dummy has to point before the head, aka
            dummy = Node(None)
            dummy.next = head

        And, we only need to advance the left pointer.
        After that, we can reverse the next part of the list, up to right, in-place.

        Finally, we update our pointer arithmetic to link the points over
        '''

        if not head or left_pos == right_pos:
            return head

        self.left = head
        right = head

        stop = False
        def recursive_swap(right, m, n):
            nonlocal stop
            
            # the base case
            if n == 1:
                return

            # move right forward
            right = right.next

            # check if we need to move left forward
            if m > 1:
                self.left = self.left.next

            # assuming m < n, guaranteed, then at some point right will pass left
            recursive_swap(right, m - 1, n - 1)

            # at this point, left and right point where we need them to
            # as we backtrack, need to make sure the problem is still valid
            if self.left == right or right.next == self.left:
                stop = True

            if not stop:
                # swap the values
                self.left.val, right.val = right.val, self.left.val

                # move the left pointer up
                self.left = self.left.next

        recursive_swap(right, left_pos, right_pos)
        return head

class Solution(object):
    def reverseBetween(self, head, m, n):

        # setup the dummy node
        dummy = ListNode(None)
        dummy.next = head
        
        # pointers to current and previous.
        # previous will let us fix the reversed links
        cur, prev = head, dummy

        for _ in range(m - 1):
            # slide both pointers forward until we reach mth node
            # if cur:
            #     cur = cur.next
            # if prev:
            #     prev = prev.next
            # # if we are sure m - 1 > number of nodes:
            cur, prev = cur.next, prev.next
        
        # reverse the needed number of nodes
        for _ in range(n - m):
            # store the current value of cur's next
            temp = cur.next
            # hop over the next element that temp points to
            cur.next = temp.next
            # reverse the link from temp to prev's next
            temp.next = prev.next
            # move up the link to prepare for the next swap
            # prev.next now points to the head of the reversed list
            prev.next = temp

        # return the head, aka can ease up thanks for the dummy
        return dummy.next
    

class Solution(object):
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # reverse lists
        l1 = self.reverseList(l1)
        l2 = self.reverseList(l2)
        
        head = None
        carry = 0
        while l1 or l2:
            # get the current values 
            x1 = l1.val if l1 else 0
            x2 = l2.val if l2 else 0
            
            # current sum and carry
            val = (carry + x1 + x2) % 10
            carry = (carry + x1 + x2) // 10
            
            # update the result: add to front
            curr = ListNode(val)
            curr.next = head
            head = curr
            
            # move to the next elements in the lists
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        if carry:
            curr = ListNode(carry)
            curr.next = head
            head = curr

        return head


class Solution(object):
    def reverseBetween(self, head, m, n):
        '''BEST AND MOST INTUITIVE SOLUTION!!!'''
        
        # pointers to previous, and head node
        p_prev, p = None, head
        for _ in range(m - 1):
            # move the pointers up
            p_prev, p = p, p.next

        print(head)

        # variables for flipping the needed nodes
        prev, cur = p, p.next
        for _ in range(n - m):
            # handy pythonic way of not needing the tmp variable
            print(prev.next)
            cur.next, prev, cur = prev, cur, cur.next

        # note: at this point, cur will be the element right after the "head" of the reverse list
        p.next = cur
        if p_prev:
            # note: p_prev was pointing right before p
            # we point it to the new head of the list
            p_prev.next, prev = prev, head
        
        return prev


class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if not head or k == 1: return head
        dummy = next_head = ListNode(None)
        dummy.next = head
        prev = curr = head

        while True:
            count = 0
            while curr and count < k:
                count += 1
                curr = curr.next
            if count == k:
                h, t = curr, prev   # assign the first node of next k-group and the first node of current k-group to h(ead), t(ail)
                for _ in range(k):   # this is NOT a standard reversing by swapping arrows between adjacent nodes
                    tmp = t.next     # instead it poplefts a node successively (ref. Campanula's comment)
                    t.next = h
                    h = t
                    t = tmp
                    # one-line implementation: t.next, t, h = h, t.next, t
                next_head.next = h   # connect the last node of the previous reversed k-group to the head of the current reversed k-group
                next_head = prev     # prepare for connecting to the next to-be-reversed k-group
                prev = curr   # head of the next yet to be reversed k-group
            else:   # curr = None and count does not reach k i.e. list is exhausted
                return dummy.next


def reorderList(self, head: Optional[ListNode]) -> None:
    """
    Do not return anything, modify head in-place instead.

    Main idea: Make a list l0 -> l1 -> l2 ... -> l_n-1 -> l_N
    Interleaved:
        l0 -> L_N -> l1 -> L_n-1 ...

    Could put everything in a list and then zip()

    Better idea:
        Split list into two halfs
        Reverse the second half
        Then, iteratively insert the nodes from the back into the front
    Have a few special edge cases:
        Making sure the lists are properly splintered after halving
        Keeping track of pointers to make sure references are accurate

    Don't return anything, modify in-place.
    """
    # base case: return the single node
    if not head: return head

    # find the middle of the list with two pointers
    slow, fast = head, head
    # when fast reaches the end of the list, slow will be at this middle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
    # at this point, we have to start from slow and reverse the rest of the list until the end(fast)
    # NOTE: slow is at the end of the first half. So, we need to reverse everything after it (slow.next -> *)
    second = slow.next
    # and, we need to "chop off" this part of the list
    slow.next = None
    
    # now we can start from the half and do a regular, iterative reversal
    prev = None
    while second:
        # store current's next
        tmp = second.next
        # reverse the link to point backwards
        second.next = prev
        # move up the pointers
        prev = second
        second = tmp

    
    # now, previous will point to the head of the other half
    # we can start flipping them
    
    # pointers to both lists
    second = prev
    first = head
    
    # we know that in odd case, the second half will be shorter. so we only have to insert its node and the last one will be ok
    while second:
        # save references to the next jump from both sides
        p1 = first.next
        p2 = second.next
        # update the pointers
        first.next = second # insert the node from the back
        second.next = p1 # make sure the new node points to rest of the front
        # update the two pointers to the references we set, and continue
        first = p1
        second = p2
        
    # nothing to return, the list is modified in place


    '''APARTE, the BRICKS FOR LINKED LISTS'''
    # 1) finding middle of the list
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next 

    # 2) List reversal
    # reverse the second half in-place
    prev, curr = None, slow
    while curr:
        tmp = curr.next
        
        curr.next = prev
        prev = curr
        curr = tmp     
    # 2a) elegant LL reversal in python
    prev, curr = None, slow
    while curr:
        # avoids tmp
        curr.next, prev, curr = prev, curr, curr.next  


    # 3) Merge two sorted lists
    first, second = head, prev
    while second.next:
        tmp = first.next
        first.next = second
        first = tmp
        
        tmp = second.next
        second.next = first
        second = tmp
    ## 3a) Elegant in python
    first, second = head, prev
    while second.next:
        first.next, first = second, first.next
        second.next, second = first, second.next


class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        
        # naive: read values into an array, check if arrays is palindrone
        vals = []
        while head:
            vals.append(head.val)
            head = head.next
        return vals == vals[::-1]
    
    
        # better: recrusive approach
        self.front = head
        
        def recursive_pal(node):
            if node:
                if not recursive_pal(node.next):
                    return False
                if node.val != self.front.val:
                    return False
                self.front = self.front.next
            return True
        
        return recursive_pal(head)

        # wild approach with fast and slow pointers, and reversing the list in place
        # also handles the case to restore the original list, in case it's important
        if head is None:
            return True

        # Find the end of first half and reverse second half.
        first_half_end = self.end_of_first_half(head)
        second_half_start = self.reverse_list(first_half_end.next)

        # Check whether or not there's a palindrome.
        result = True
        first_position = head
        second_position = second_half_start
        while result and second_position is not None:
            if first_position.val != second_position.val:
                result = False
            first_position = first_position.next
            second_position = second_position.next

        # Restore the list and return the result.
        first_half_end.next = self.reverse_list(second_half_start)
        return result    

    def end_of_first_half(self, head):
        fast = head
        slow = head
        while fast.next is not None and fast.next.next is not None:
            fast = fast.next.next
            slow = slow.next
        return slow

    def reverse_list(self, head):
        previous = None
        current = head
        while current is not None:
            next_node = current.next
            current.next = previous
            previous = current
            current = next_node
        return previous


class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        '''Nice and clean solution, where we don't care about "fixing" the list.'''
        slow, fast, prev = head, head, None
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        prev, slow, prev.next = slow, slow.next, None
        while slow:
            slow.next, prev, slow = prev, slow, slow.next
        fast, slow = head, prev
        while slow:
            if fast.val != slow.val: return False
            fast, slow = fast.next, slow.next
        return True


        ## alternative with extra checks/catches on fast
        rev = None
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            rev, rev.next, slow = slow, rev, slow.next
        if fast:
            slow = slow.next
        while rev and rev.val == slow.val:
            slow = slow.next
            rev = rev.next
        return not rev


        ## reversing the list after we're done, in case
        rev = None
        fast = head
        while fast and fast.next:
            fast = fast.next.next
            rev, rev.next, head = head, rev, head.next
        tail = head.next if fast else head
        isPali = True
        while rev:
            isPali = isPali and rev.val == tail.val
            head, head.next, rev = rev, head, rev.next
            tail = tail.next
        return isPali


        '''NOTE: commented solution'''
        # rev records the first half, need to set the same structure as fast, slow, hence later we have rev.next
        rev = None
        # initially slow and fast are the same, starting from head
        slow = fast = head
        while fast and fast.next:
            # fast traverses faster and moves to the end of the list if the length is odd
            fast = fast.next.next
            
            # take it as a tuple being assigned (rev, rev.next, slow) = (slow, rev, slow.next), hence the re-assignment of slow would not affect rev (rev = slow)
            # rev, rev.next, slow = slow, rev, slow.next is equiv to::
            # tmp = rev
            # rev = slow
            # slow = slow.next
            # rev.next = tmp
            rev, rev.next, slow = slow, rev, slow.next
        if fast:
        # fast is at the end, move slow one step further for comparison(cross middle one)
            slow = slow.next
        # compare the reversed first half with the second half
        while rev and rev.val == slow.val:
            slow = slow.next
            rev = rev.next
        
        # if equivalent then rev become None, return True; otherwise return False 
        return not rev



   
        
def hasCycle(self, head: Optional[ListNode]) -> bool:
    '''Find whether a linked list has a cycle.'''

    '''
    Trivial to do with a hashset, just check for node membership
    '''
    if not head: return None
    # with hash set
    visited = set()
    while head:
        # if we haven't seen it, add it to the set
        if head not in visited:
            visited.add(head)
        else:
            # already visited, aka we found a cycle
            return True
        head = head.next # move on up
    return False # visited all nodes and hit Null without a cycle (redundant)


    '''Tortoise and Hare, two pointer'''
    hare, tortoise = head, head

    # NOTE: this is the tricky part. We need to be able to double-hop the fast pointer
    while hare and hare.next:
        tortoise = tortoise.next
        hare = hare.next.next
        if hare is tortoise:
            return True

    # fast reached the end of the list first without hitting a cycle
    return False


def mergeLL(l1, l2):
    '''Merging two sorted LinkedLists'''

    # start with a dummy node to avoid weird edge cases
    dummy = ListNode()
    tail = dummy

    # need to pick form either list while both are full
    while l1 and l2:

        # if the first list's element is smaller, add it to our running list
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next

        # else, add the second element's list
        else:
            tail.next = l2
            l2 = l2.next

        # update tail pointer always, regardless of which lists' element was picked
        tail = tail.next

    # one of the lists could still have values 
    tail.next = (l1 or l2)

    # return the next value of our dummy head, which will have the first element added
    return dummy.next


    '''Nice resursive solution for reference'''
    def mergeTwoLists(self, l1, l2):
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2


def mergeKLists(lists):
    '''Leveraged the simple solution of merging two linked lists, and does a version of merge sort.

    Instead of merging all lists all the time, it does so in groups of two''''
    # handle the edge case
    if not lists: return None

    # while we have lists to process
    while len(lists) > 1:
        # the grouped, merged lists
        merged = []

        # group the pairs
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i+1] if i + 1 < len(lists) else None
            merged.append(mergeLL(l1, l2))

        # update our lists
        lists = merged

    # return the first, aka totally merged list
    return lists[0]


    '''Brute force solution, gathering all nodes, sorting, then creating the final list'''
    self.nodes = []
    head = point = ListNode(0)
    for l in lists:
        while l:
            self.nodes.append(l.val)
            l = l.next
    for x in sorted(self.nodes):
        point.next = ListNode(x)
        point = point.next
    return head.next

    '''With a priority queue, for reference'''
    from queue import PriorityQueue
    # need this class because Node does not implement LT
    class Wrapper():
        def __init__(self, node):
            self.node = node
        def __lt__(self, other):
            return self.node.val < other.node.val
            
    head = point = ListNode(0)
    q = PriorityQueue()
    for l in lists:
        if l:
            q.put(Wrapper(l))
    while not q.empty():
        node = q.get().node
        point.next = node
        point = point.next
        node = node.next
        if node:
            q.put(Wrapper(node))
    return head.next


def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    '''Another two-pointer with dummy head solution.

    Main idea:
        Start left pointer at dummy node
            Set dummy node's next to be the head

        Start right pointer at the original head
        Advance the right pointer "n" steps

        Then, until the right pointer is none:
            Advance both left and right pointers

        At this point, the left pointer is right before the n-th node to delete.

        Hop over the connection [a -> b -> c] --> [a -> c]
            left.next (b) = left.next.next (c) # => a-c

        Return the "next" of the dummy node which will point to the new, shorter list.
    '''
    
    # create the dummy node
    dummy = ListNode()
    dummy.next = head
    # start our left pointer at the dummy
    left = dummy
    
    # start right pointer at the head of the list
    right = head
    
    # advance right pointer n-steps
    for _ in range(n):
        right = right.next
        
    while right:
        right = right.next
        left = left.next
        
    # at this point, we're at the node to delete
    left.next = left.next.next
    
    # return the next of the dummy node, which will point to list with element removed
    return dummy.next
    

        

def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    
    # base case: return the single node
    if not head: return head
    
    # setup pointers to current head and new head
    cur = head
    prev = None # note: important to start as None, else we get a dummy tail node

    # while there is a list to process...
    while cur:

        # store the next-pointer of the current node we're about to break
        tmp = cur.next

        # point the next link to the new tail
        cur.next = prev

        # update the pointers
        prev = cur # straightforward
        cur = tmp # the previous next node, who's chain we broke

    # return the new head, which will start at the tail of the old list
    return prev


    '''Recursive implementation
    Start from subproblem of reversing a last node that points to None, and work up
    '''
    if not head:
        return None # the base case, there is nothing left to process

    # start the new head pointing to the current node
    newHead = head

    # check if there is a subproblem to solve
    if head.next:

        # reverse the remainder of the list
        newHead = self.reverseList(head.next)

        # swap the links # [a -> b -> c] --> [a -> b -> a]
        head.next.next = head

    # make the next head point to null: [a -> b -> a] --> [a -> None]
    head.next = None

    return newHead
    '''NOTE: stack of function call for test case [1,2]

        # they start of equal
        PRE idx=0, head=ListNode{val: 1, next: ListNode{val: 2, next: None}}, new_head=ListNode{val: 1, next: ListNode{val: 2, next: None}}

        # then, because head.next exists, we go into the first recursive call
        PRE idx=1, head=ListNode{val: 2, next: None}, new_head=ListNode{val: 2, next: None}

        # because in this call head.next does not exist, we return immediatelt
        POST idx=1, head=ListNode{val: 2, next: None}, new_head=ListNode{val: 2, next: None}

        # now we are popped up the recursion stack
        # the new head points to `head.next`
        # NOTE: we get to a cycle here: [A -> B -> A]
        # we break this by setting head's next to None: [A -> None]
        POST idx=0, head=ListNode{val: 1, next: None}, new_head=ListNode{val: 2, next: ListNode{val: 1, next: None}}
    '''

    '''Another recursive implementation, shorter and more direct'''
    if node is None or node.next is None:
        # either head is empty or there is no next link
        return node

    # if there are nodes to process, reverse them
    new_head = reverse(node.next)
    # update the next node's value and break the cycle
    node.next.next = node
    node.next = None
    return new_head

