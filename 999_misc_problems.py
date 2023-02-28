'''Problems that don't fit neatly into other categories. 
'''

class Solution:
    '''Bucket solution using complement buckets
    '''
    def numPairsDivisibleBy60(self, time):
        c = [0] * 60
        res = 0
        for t in time:
            res += c[(60 - t % 60) % 60] # c[-t % 60]
            c[t % 60] += 1
        return res



def isValid(self, s: str) -> bool:
    '''Classic problem. 

    Main logic uses a stack, building up from the fact that valid sub-expressions always builds up from a single set of parentheses.

    The main trip up is in edge cases: 
        whether the stack is empty at the end
        when we reach a closing bracket, is there anything on the stack? (the only place a closer could be)
        what if we are given no parentheses, or a single one?

    NOTE: above is technically O(n/2) since we don't add the closer to the stack.
    '''
    # edge cases: empty string or a single parentheses -> immediately invalid
    if len(s) <= 1: return False
    
    # stores the values seen so far
    stack = []
    
    # map from opening to closing brackets
    open2close = {
        '{': '}',
        '(': ')',
        '[': ']',
    }
    
    # step through all parentheses
    for c in s:

        # if we are dealing with an opening parentheses...
        if c in open2close:
            stack.append(c)

        # this is a closing parentheses, check if we can close it
        else:
            # if there is something on the stack, we could possibly close this one.
            if stack:
                # grab the most recent parentheses
                prev = stack.pop()
                # if the types of parentheses don't match, we can't close, aka invalid
                if open2close[prev] != c:
                    return False
            
            # if the stack is empty, we couldn't possibly close this parentheses, aka invalide
            else:
                return False

            # NOTE: the if/else above can be cleaned up:
            # # alternative
            # if not stack or open2close[stack.pop()] != c:
            #     return False

    # at the end, every opener should have found a matching closer.
    # if this was not the case, we have a dangling opener, thereforce incorrect
    return not stack


    '''Simpler version adding everything to the stack
    
    NOTE: this cleaner version requires open2close
    '''
    stack = []
    close2open = {v: k for k,v in open2close.items()}
    for c in s:
        if c in close2open:
            if stack and stack[-1] == close2open[c]:
                stack.pop()
            else:
                return False
        else:
            stack.append(c)
    
    return not stack




'''Implementing a FIFO queue with two stacks
The main idea is keeping two stacks, and shuffling items from one to the other to keep the "FO" ordering when popping.

When adding a new element:
    Move over all elems in s1
    Append new elem to empty s1
    Move back over all elems from s2

    Example: with elements so far
    pushing (b, c, d)...
    -> push(b)
        [], []
        [b], []
    -> push(c)
        [], [b] # move all elems to other stack(in this case, only b)
        [c], [b] # added new element
        [c, b], [] # popped b from other stack, appended to first stack
        # now, poppping from s1 would rightly return the firstly added b
    -> push(d)
        [], [b, c]
        [d], [b, c]
        [d, c, b], [] # if we popped everything, would get correct queue FIFO order: b, c, d

    # driving the point home, with slightly different order
    [b, c, d], []
    -> push(a) (assuming new order of push: d, c, b)
    [], [d, c, b]
    [a], [d, c, b]
    [a, b, c, d]
    -> popall(): [d, c, b, a], as expected
'''
class MyQueue:

    def __init__(self):
        self.s1 = []
        self.s2 = []

    def push(self, x):
        # move elements from one to the other so element is placed "on top"
        while self.s1:
            self.s2.append(self.s1.pop())
        self.s1.append(x)
        while self.s2:
            self.s1.append(self.s2.pop())

    def pop(self):
        return self.s1.pop()

    def peek(self):
        return self.s1[-1]

    def empty(self):
        return not self.s1
        
    '''There is an optimization, which doesn't move the elements every time.
    Rather, it only adds to s1.
    Then, when a pop is requested, we move over all elements from s1 to s2.
        Thanks to the inverse ordering, the top element of s2 will be what we're after

    Then, peek/pop continue returning from s2 which has the correct FIFO ordering until its empty
    Then, once s2 is empty, we can resume the process with incoming s1
        The values so far in s1, that will eventually be popped/appended to s2, will retain the original, intended FIFO ordering.

    AMORTIZED: instead of moving over each elem to s2, and back again
                Move chunks over, as needed, and leverage the implicit FIFO ordering 
    '''
    def push(self, x):
        # O(1), not moving all elems each time
        self.s1.append(x)

    def pop(self):
        self.peek()
        return self.s2.pop()

    def peek(self):
        # every "n" items, we have to do O(n) work to move them over
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        # every (n-1) items, we only have to do O(1) lookup
        # -> (n + 1 + 1 ....) / n = (1 + 1/n + 1/n + 1/n)
        # 1 + sum_(n-1)(1/n), lim n -> inf = 0
        # 1 + sum_(n-1)(0) ~= 1
        return self.s2[-1]        

    def empty(self):
        return not self.s1 and not self.s2
        
                



'''Majority element: trivial with a hashmap.

1) count up the elements
2) run through, first time we find one that meets the floor criteria, return it

Can combine 1 and 2 in the same loop: check for 2) right after we update count 1).

'''
class Solution:
    def majorityElement(self, nums: List[int]) -> int:

        '''Two-pass, loops'''
        floor_thr = len(nums) // 2
        cnts = {}
        for n in nums:
            cnts[n] = 1 + cnts.get(n, 0)
            
        for n in cnts:
            if cnts[n] > len(nums) // 2:
                return n 

        '''One pass, check element at the same time in one loop'''
        floor_thr = len(nums) // 2
        cnts = {}
        for n in nums:
            cnts[n] = 1 + cnts.get(n, 0)
            if cnts[n] > floor_thr:
                return n
            

        '''Notes on a clever solution: the Boyer-Moore voting algorithm
            Intuition
            If we had some way of counting instances of the majority element as +1+1 and instances of any other element as -1−1, summing them would make it obvious that the majority element is indeed the majority element.

            Algorithm
            Essentially, what Boyer-Moore does is look for a suffix sufsuf of nums where suf[0]suf[0] is the majority element in that suffix. To do this, we maintain a count, which is incremented whenever we see an instance of our current candidate for majority element and decremented whenever we see anything else. Whenever count equals 0, we effectively forget about everything in nums up to the current index and consider the current number as the candidate for majority element. It is not immediately obvious why we can get away with forgetting prefixes of nums - consider the following examples (pipes are inserted to separate runs of nonzero count).

            [7, 7, 5, 7, 5, 1 | 5, 7 | 5, 5, 7, 7 | 7, 7, 7, 7]

            Here, the 7 at index 0 is selected to be the first candidate for majority element. count will eventually reach 0 after index 5 is processed, so the 5 at index 6 will be the next candidate. In this case, 7 is the true majority element, so by disregarding this prefix, we are ignoring an equal number of majority and minority elements - therefore, 7 will still be the majority element in the suffix formed by throwing away the first prefix.

            [7, 7, 5, 7, 5, 1 | 5, 7 | 5, 5, 7, 7 | 5, 5, 5, 5]

            Now, the majority element is 5 (we changed the last run of the array from 7s to 5s), but our first candidate is still 7. In this case, our candidate is not the true majority element, but we still cannot discard more majority elements than minority elements (this would imply that count could reach -1 before we reassign candidate, which is obviously false).

            Therefore, given that it is impossible (in both cases) to discard more majority elements than minority elements, we are safe in discarding the prefix and attempting to recursively solve the majority element problem for the suffix. Eventually, a suffix will be found for which count does not hit 0, and the majority element of that suffix will necessarily be the same as the majority element of the overall array.
        '''
        count = 0
        candidate = None

        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)

        return candidate
        


def countPrimes(self, n):
    """
    :type n: int
    :rtype: int

    Count primes using the Sieve of Erastothenes
    """
    if n < 2:
        return 0
    s = [1] * n
    # 0 and 1 are not prime
    s[0] = s[1] = 0
    # for numbers starting from 2
    for i in range(2, int(n ** 0.5) + 1):
        # if this value is "on"
        if s[i] == 1:
            # mark all integer multiples of this number as False
            for j in range(i * i, n, i):
                s[j] = False
    # only prime numbers will have been left untouched
    return sum(s)