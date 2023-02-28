'''Climbing stairs, found the perfect way to get it now.
With special handling for n <= 3
'''
class Solution:
    def climbStairs(self, n: int) -> int:
        
        # base cases: 1-3 steps
        if n <= 3: return n
        
        # ways to climb one stair
        to_first = 1
        # ways to climb two stairs
        to_second = 2
        # total ways
        ways = 0
        for _ in range(n - 2): #already climbed the first two
            ways = to_first + to_second
            # move up the step
            to_first = to_second
            # set the new total ways
            to_second = ways
            
        # return the total ways to climb
        return ways


def word_break(s, wordDict):
    '''
    Another dynamic programming.
    This time, we are trying to see if we can break up the string `s` into equal chunks with words made up from `wordDict`.

    The brute force approach is to check every substring in the sequence s, check if it is a word, and move the pointer forward.

    But, if the words are shorter than the string s, we can do better by comparing the words directly. 

    For a dynamic programming solution, we can start from the very back.
    Why? If we somehow reached the end of the string, that means we were able to reach the end. 

    So then we start working our way backwards, and check if there are words that fit/split the string so far.
    '''

    # whether we can insert a word at index i
    splits = [False for _ in range(len(s) + 1)]
    # if we reached the end of the word, we know it's splittable
    splits[-1] = True

    # start working our way back
    for i in range(len(s)-1, -1, -1): # reversed(range(len(s)))
        # check every word
        for w in wordDict:
            # check if splicing in this word would keep us in bounds
            if i + len(w) <= len(s):
                # we are in bound! so, if the word matches ours so far...
                if s[slice(i, i+len(w))] == w:
                    # check if we could split again after this word
                    splits[i] = splits[i + len(w)]

            # if we met both conditions in the loop above, we found a good word. Even if another word is possible, we don't need to check it.
            if splits[i]: break

    # return the first index, which will be True if we can reach the end from i==0 using the words in wordDict
    return splits[0]

    '''Top-down approach'''
    ok = [True]
    max_len = max(map(len, words + ['']))
    words = set(words)
    for i in range(1, len(s) + 1):
        # checks all words with generators, terminates when a single true is hit
        ok += any(ok[j] and s[j:i] in words for j in range(max(0, i-max_len), i)),
    return ok[-1] # will be true if we were able to split to the end of the word

    '''Incredible approach using a Trie, need to learn what this is'''
    class TrieNode:
        def __init__(self):
            self.isWord = False
            self.child = defaultdict(TrieNode)
        
        def addWord(self, word):
            cur = self
            for c in word:
                cur = cur.child[c]
            cur.isWord = True

        root = TrieNode()
        for word in wordDict:
            root.addWord(word)
            
        n = len(s)
        dp = [False] * (n+1)
        dp[n] = True
        
        for i in range(n-1, -1, -1):
            cur = root
            for j in range(i+1, n+1):
                c = s[j-1]
                if c not in cur.child: break  # s[i:j] not exist in our trie
                cur = cur.child[c]
                if cur.isWord and dp[j]:
                    dp[i] = True
                    break
        
        return dp[0]




def maxRobbery(houses):
    '''
    Another problem that can be converted from top-down recursive to bottom-up DP.
    This one unfortunately has no greedy solution.

    Great general pattern for dynamic programming.
    We can rob the current house or the next, but we can't rob adjacent houses.
    With two houses, it's easy: rob the greatest one
    With three, it become interesting:
        - Do we rob the first, and third
        - Do we rob the second?
    More generally:
        - Is the optimal solution robbing the current house (i), and robbing the *rest + 1* of the array: (i+2)
        - Is the optimal solution skipping the current house (i), and robbing the *rest* of the array?
    '''

    '''Naive solution with memoization. O(N) time and space thanks the memoization, else 2^n'''
    def __init__(self): self.memo = {}
        
    def rob(self, nums: List[int]) -> int:
        
        '''Naive recursive solution with memoization'''
        def robFrom(i):
            # base case: we overshoot the array
            if i >= len(nums):
                return 0
            # look up memoized answer or compute it
            if i in self.memo:
                return self.memo[i]
            else:
                self.memo[i] = max(nums[i] + robFrom(i+2), robFrom(i+1))
            
            return self.memo[i]
        
        return robFrom(0)


    '''Replacing recursion above with tabulization'''
    # setup DP array
    dp = [0] * (len(nums) + 1)
    # two base cases: no houses left to rob. Or only one house left, so rob it.
    dp[-1], dp[-2] = 0, nums[-1]
    
    # iterate back, answering the main robbing question
    for i in range(len(nums)-2, -1, -1):
        dp[i] = max(dp[i+1], dp[i+2] + nums[i])
        
    return dp[0]


    '''Optimized DP solution starting from the back of the array'''
    # space optimized DP with base cases (assuming we reached the end)
    cur_house = 0         # no houses left at the end
    prev_house = nums[-1] # rob the last house 
    
    for i in range(len(nums)-2, -1, -1):
        
        cur_best = max(prev_house, cur_house + nums[i])
        
        # move backward
        cur_house = prev_house
        prev_house = cur_best
        
    return prev_house
    
    '''Optimized DP solution starting from the front of array.'''
    if not nums: return None
    if len(nums) <= 2: return max(nums)

    rob1, rob2 = 0, 0 # [rob1, rob2, ...]
    # walk through the houses
    for n in nums:
        local_best = max(rob2, rob1 + n) # rob current house and skip the next, or rob the second?
        # move to the next house
        rob1 = rob2
        # what best, local value did we end up with?
        rob2 = local_best
    # return the max we saw at the end of the array
    return rob2 

    '''**Saving ourselves a tiny bit of loop, mimicking starting from the back'''
    if not nums: return None
    if len(nums) <= 2: return max(nums)
    
    # base cases: don't rob the first house, rob the second
    rob1, rob2 = 0, nums[0] # [rob1, rob2, ...]
    # walk through the houses
    for n in nums[1:]:
        # to start, thie immediately check (i+2 + i, vs i + 1)
        local_best = max(rob2, rob1 + n) # rob current house and skip the next, or rob the second?
        # move to the next house
        rob1 = rob2
        # what best, local value did we end up with?
        rob2 = local_best
    # return the max we saw at the end of the array
    return rob2 

    '''Commented version of the above solution'''
    # Sanity check for an empty neighborhood 
    if not nums: return 0

    # Two base cases:
    rob1 = 0  # start at the beginning, we have not robbed any houses
    rob2 = nums[0]  # however, we could choose to rob the first house
    
    # Visit the remaining houses, after the first one
    # NOTE: In the case of only one house, this `for` statement never runs and the best we could do is simply robbing the first house
    for n in nums[1:]:
        
        # What's better:
        #  1) Rob the previous house and skip the current one: `rob2`
        #  2) Rob the current house and keep deciding: `rob1 + n`
        # NOTE: On the first iteration, this simply compares the first two houses. Afterward, it is the recurrence relationship
        local_best = max(rob2, rob1 + n)  
        
        # move up our robber and the value we've seen so far
        # NOTE: On the first iteration, this sets our "previous" house as the first one `nums[0]`. Afterward, it holds the running value of the recurrent relationship.
        rob1 = rob2
        
        # what was the best, local robbery decision so far?
        # NOTE: After the first iteration, this variable stores the max value of the recurrence relationship so far
        rob2 = local_best

    # return the max, total value we found after visiting all houses
    return rob2


def maxRobberyTwo(houses):
    '''Same as above, with a new condition that the houses at the edges are linked.
    
    Key insight: we can re-use our solution from HR1, but we need to call it twice, on the trailing/lagging parts of the array.
    Also, we need special care for arrays with a single elements
    '''
    # if not houses: return None
    # if len(houses) == 1: return houses[0] # can get finnicky if house connections viewed as graph
    # node[0].neighbors == node[0] => robbery singularity
    # python base case checking
    if len(houses) <= 1:
        try:
            return houses[0]
        except IndexError:
            return 0

    def rob_helper(nums):
        "Solution to House Robber One."
        prev_rob, cur_rob = 0, nums[0]
        for n in nums[1:]:
            cur_val = max(cur_rob, prev_rob + n)
            prev_rob, cur_rob = cur_rob, cur_val
        return cur_rob

    return max(rob_helper(houses[1:]), rob_helper(houses[:-1])    


def decodeWays(s):
    '''How many ways can we decode the numerical string `s`, if the alphabet is encoded as:
    a -> 1
    ....
    z -> 26

    Can take either one or two characters at a time, leading to a potentially large decision tree.
    But, each problem depends on the solution to its subproblem, so we could start from the "end", and work our way up. By the time we reach the larger problem, the smaller subproblems it depends on will have been solved. 

    Few edge cases:
        Numbers cannot start with 0
        If 2-digit numbers start with 1, any character can follow them.
        If 2-digit number start with 2, chars between 0 and 6 can follow them.

    Brute-force:
        recursive with memorization
    '''

    '''Brute force memo'''
    # base case: we got to the end of the string
    dp = {len(s) : 1}

    def dfs(i):
        if i in dp:
            return dp[i]
        if s[i] == '0':
            return 0

        # decode the single character
        res = dfs(i+1)

        # check if we can dual-digit decode
        if i + 1 < len(s):
            # check if this would be a valid 2-digit value
            if s[i] == '1' or (s[i] == '2' and ord('0') <= ord(s[i+1]) <= ord('6')): 
                res += dfs(i+2)

        # cache the solutiuon
        dp[i] = res

        # return the solution
        return res

    # start from the beginning
    return dfs(0)


    '''Memo converted to bottom-up DP.
    Still using the cache, base cases are slightly similar.'''
    dp = {len(s): 1}

    # walk from the back
    for i in range(len(s)-1, -1, -1):
        # check the base cases
        if s[i] == '0':
            dp[i] = 0
        # populate with the already-solved subproblems
        else:
            dp[i] = dp[i+1]

        # check valid two-digit decodes
        if i + 1 < len(s):
            # check if this would be a valid 2-digit value
            if s[i] == '1' or (s[i] == '2' and ord('0') <= ord(s[i+1]) <= ord('6')): 
                dp[i] += dp[i+2]

    return dp[0]


def uniquePaths(m, n):
    '''
    How many ways can we reach the bottom-right cell of a grid, starting from the top-left?
    Assuming we can only take steps to the right, and down?

    There is an intuitive, recursive solution:
        From the first row and column, we can only move right and down, respectively.
            I.E all steps of 1.
        For the inner rows and columns, we have to work our way in from the first row/col
            For each inner-inner row, we can find our way in increasingly more ways from the inner row (and col)

    recursive(m, n):
        if 0 in (m,n):
            return 0
        if 1 in (m,n):
            return 1
        return recursive(m-1, n) + recursive(m, n-1)

    Or, similar to the Longest Common Subseq, we can populate a 2D grid and DP fill the sub-problems.

    Lastly, again similar to LCS, we can save space by noticing that only-ever two columns or rows are needed.'''

    # dp solution populating the grid
    d = [[1] * n for _ in range(m)]

    # start from the inner rows, populate the ways we can reach them 
    for col in range(1, m):
        for row in range(1, n):
            # explicit iterative version of what the recursive approach
            d[col][row] = d[col - 1][row] + d[col][row - 1]

    # sum up the ways we were able to reach the cells directly above and to the left of the solution
    return d[m - 1][n - 1]

    '''Space-saved solution from neetcode, similar to above. Only using two rows'''
    # start with the bottom row
    row = [1] * n 
    
    # iterate through the remaining rows
    for i in range(m - 1):
        # create the current row
        # NOTE: can avoid creating this at every loop, if we define above. BUT! Need to make sure re-assignment is proper
        new_row = [1] * n
        # add up the adjacent ways to reach a given cell
        # at the end, the zeroth spot of this new row holds the final answer
        for j in range(n-2, -1, -1): # n-2 to skip the last entry which will always be 1
            new_row[j] = new_row[j + 1] + row[j] # (right, down)
            
        # prepare for the next row
        row = new_row
        # # NODE: to avoid creating new_row at every loop:
        # row, new_row = new_row, row
        
    # return the final, stored answer at the top
    return row[0]


    '''We can save even more space, avoiding the second array(!!!)'''
    # if there are more rows than columns, swap them
    if m > n:
        m, n = n, m

    # the single row we need
    r = [1] * m
    for _ in range(1, n):
        for i in range(1, m):
            # NOTE: row above started as all 1s. 
            # we can accumulate the value directly, and it all tracks in here
            r[i] += r[i - 1] # r[i] + r[i - 1]
    return r[-1]

    '''A nice commented Java solution'''
    public int uniquePaths(int m, int n) {
        int[][] grid = new int[n][m];
        
        for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (i == 0) grid[0][j] = 1;
            if (j == 0) grid[i][0] = 1;
            if (i != 0 && j != 0) {
            int up = grid[i - 1][j];
            int left = grid[i][j - 1];
            grid[i][j] = up + left;
            }
        }
        }
        return grid[n - 1][m - 1];
    }


def jumpGame(nums):
    '''Each value in an array tells us the maximum size of a "jump" we can take from that index.

    The question is: starting from the beginning, is there a way we can reach the beginning? 

    There is a DP/cached solution that is based on a branching decision tree.
    Our stopping condition is when we reach a value/node where it's impossible to get to the end.

    There is, however, an O(N) solution starting from the back.

    Assume we made it to the end. Then, working backwards:
        from i - 1, could we make it to the end? If yes, move the goalpost back.

    At the end, the goalpost should be at the front. If it's not, that means we were not able to reach it!

    Note: this graph has a good, alternative back-track and recursive solution that's good to flesh out.
    Another one of those we should do in NB, lots to learn:
        https://leetcode.com/problems/jump-game/solution/
    '''
class Solution:
    def canJump(self, nums: List[int]) -> bool:

        # top-down memoization
        memo = {}
        def dfs(pos):
            if pos in memo:
                return memo[pos]

            if pos == len(nums) - 1:
                return True
            if pos >= len(nums):
                return False

            # check all hops from this position
            # small optimization: take the largest hop you can
            for i in range(1, nums[pos] + 1)[::-1]:
                if dfs(pos + i):
                    memo[pos + 1] = True
                    return memo[pos + 1]
            
            memo[pos] = False
            return memo[pos]

        can_reach = dfs(0)
        return can_reach 

    # greedy from the front
    m = 0 # tracks the maximum index we can jump to so far
    for i, n in enumerate(nums):
        if i > m:
            return False
        m = max(m, i+n)
    return True

    # greedy from the back
    goal = len(nums) - 1
    for i in range(len(nums)-1, -1, -1):
        if i + nums[i] >= goal:
            goal = i
    return goal == 0


def climbStairs(n):
    '''
    Dynamic programming approach.
    Key point: look at the decision tree, and where the repeated work is. 
    There is a lot of repeated work starting from the top. 
    It's better to start at the end, since the solution at the end will always look the same.

    We are basically doing the Fibonacci sequence.

    There are `1` ways to reach the end from the last two positions.
    Then, we can recursively start working back and summing the solutions for the sub-problems. 
    '''
    # start from the two last slots in the array
    # NOTE: this works because "climbing the stairs" starts from 0, ground floor
    # our F[0] = 0 is implicit
    one, two = 1, 1
    
    # process the rest of the cells 
    for i in range(n - 1):
        # store one since we are modifying it, but need the value
        tmp = one
        # update number of way to reach the current slot
        one = one + two
        # move the other pointer back 
        two = tmp
        # # NOTE: can do this in one-liner swap assignment, but have to be careful
        # two, one = one, two + one # make sure two is updated "first"

    return one 

    # cleaner solution, starting truly from 0
    # explicit edge cases
    # NOTE: we never use the first slot, == 0 
    if n in (0,1,2): return n # if n <= 2: return n 
    dp = [0]*(n+1) # considering our starting point, aka "no stairs to climb!!" steps we need n+1 places
    # the first step could be loosely seen as the base case
    # then, set the appropriate number of steps
    dp[1]= 1
    dp[2] = 2
    for i in range(3, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

    '''**ALTERNATIVE DP SOLUTION'''
    if n <= 3: return n

    dp = [1]*(n) # considering our starting point, aka "no stairs to climb!!" steps we need n+1 places
    # the first step could be loosely seen as the base case
    # then, set the appropriate number of steps
    dp[1] = 2
    for i in range(2, n):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[-1]

    '''THIS IS THE ONE STAIR CLIMB:'''
    if n <= 3: return n
    ways = 0
    ways_to_first = 1 
    ways_to_second = 2
    for _ in range(n-2): # already counted the first two steps: consider the rest
        ways = ways_to_first + ways_to_second
        ways_to_first = ways_to_second
        ways_to_second = ways
    return ways


    '''We never use the first entry, can reduce some overhead in the loop'''
    if n <= 3: return n # IMPORTANT BASE CASE CHECK
    ways_to_climb = 0
    # ways to climb the base cases: starting from i = 3
    two_steps_below_current = 1  # 2 steps below 3 - ways to take 1 step: 1 #< NOTE: this refers to the first step base case!
    one_step_below_current  = 2  # 1 step below 3 - ways to take 2 steps: 2 #< NOTE: this refers to the second step base case!
    for i in range(3, n + 1):
        # find the combined number of ways to reach this step
        ways_to_climb = two_steps_below_current + one_step_below_current
        # we're going one step up to i + 1
        # so what was the num ways for 1 step below (i - 1)
        # is now the num ways for 2 steps below (i + 1 - 2 = i - 1)
        two_steps_below_current = one_step_below_current
        # what was the number of ways for i
        # is now the num ways for 1 step below (i + 1 - 1 = i)
        one_step_below_current = ways_to_climb

    return ways_to_climb

    '''Clean python Fib for reference'''
    a = b = 1
    for _ in range(n):
        a, b = b, a + b
    return a




def coinChange(coins, amount):
    '''
    Dynamic programming again.
    Look at the decision tree.

    Starting from amount and branching out with every coin, see what sub-problem we're left with.
    Note: this is an unbounded knapsack, since we can re-use the coins.
    Eventually, we notice that there's a lot of repeat work. And we only care about leaf nodes with values of 0, meaning we found exact change.

    Can either do a Depth-First brute force (backtrack?) solution, or a bottom-up dynamic programming approach.
    '''
    # compute the sentinel value, what should be an impossible way of reaching the amount
    clip = amount + 1
    
    # array for computed values
    res = [clip for _ in range(clip)]

    # we know the "base" case of 0, we need 0 coins to reach it (redundant)
    res[0] = 0

    # solve the remaining sub-problems for number of ways to reach amount `a`
    for a in range(1, clip):
        # we need to check every coin option
        for c in coins:
            # if we don't go negative by using this coin (i.e. overdraw)
            if a - c >= 0:
                # two options, either the current value is a minimum (aka we could take a single coin)
                # or, we take the current coin, and the solution for the leftover/remaining change
                # note: we are building from 0-change left to reach amount, so at `a` we'll have the results from `a-c`, and a-c is in bounds since we checked it's non-negative
                res[a] = min(res[a], 1 + res[a - c])

    # need to catch one edge case. if the amount did not change, that means we couldn't find a combination of change. In that case return the sentinel value of -1
    # more specifically, the min() condition always reached clip, since the RHS was `1 + clip`
    return -1 if res[amount] == clip else res[amount] #(res[amount], -1)[res[amount] == clip]


def findLIS(nums):
    '''
    Find the longest increasing subsequence. 
    A subsequence can be built by removing or deleting certain elements in an array.

    Note: the official answer on leetcode for this question is excellent
    Gives a great breakdown of DP approach:
    https://leetcode.com/problems/longest-increasing-subsequence/solution/
    
    This is another dynamic programming question, which we can tell by the fact that we're asked for a min/max of a sequence. 

    We can start with a depth-first, brute force solution.
    At each element, we can either include or not include it in our sequence. 
    Whether we do or don't depends on the previous value, and whether it's smaller than the current (aka including the current would keep the sequence increasing)

    As usual in this decision tree we spot lots of repeated work. 
    The repeated work + binary decision per element leads to O(2 ^ n), brutal.
    By taking a bottom-up approach and caching the intermediate outputs (also known as memoization), we can find a more efficient solution.
        The most intuitive solutions are O(n ^ 2). There is an nlogn solution via binary search that's harder to find (although it isn't the biggest leap once you understand the second solutions)
    
    For each element, the base case is simply including that value.
    So we can start with all sequences being of length one. 
    '''
    # base case: we take each element as its own longest sequence 
    res = [1 for _ in range(len(nums))]

    # Approach 1a) start from the back
    for i in range(len(nums)-1, -1, -1):
        # check all of the forward elements
        for j in range(i+1, len(nums)):
            # make sure this previous number would keep the sequence increasing
            if nums[i] < nums[j]:
                # compute the LIS for current element based on the already computed ones
                res[i] = max(res[i], 1 + res[j])
    
    # return the longest subsequence we found
    return max(res)

    # # Approach 1b) we can also start from the front
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        # compute all numbers up to i (aka the previous vals)
        for j in range(i): # implicitly up to i
            # now we need to check the current number is larger than the previous ones
            if nums[i] > nums[j]:
                # recurrent relation remains the same
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


    # # Approach 2) intelligently builD the longest subsequence
    '''
    As stated in the previous approach, the difficult part of this problem is deciding if an element is worth using or not. Consider the example nums = [8, 1, 6, 2, 3, 10]. Let's try to build an increasing subsequence starting with an empty one: sub = [].

    At the first element 8, we might as well take it since it's better than nothing, so sub = [8].

    At the second element 1, we can't increase the length of the subsequence since 8 >= 1, so we have to choose only one element to keep. Well, this is an easy decision, let's take the 1 since there may be elements later on that are greater than 1 but less than 8, now we have sub = [1].

    At the third element 6, we can build on our subsequence since 6 > 1, now sub = [1, 6].

    At the fourth element 2, we can't build on our subsequence since 6 >= 2, but can we improve on it for the future? Well, similar to the decision we made at the second element, if we replace the 6 with 2, we will open the door to using elements that are greater than 2 but less than 6 in the future, so sub = [1, 2].

    At the fifth element 3, we can build on our subsequence since 3 > 2. Notice that this was only possible because of the swap we made in the previous step, so sub = [1, 2, 3].

    At the last element 10, we can build on our subsequence since 10 > 3, giving a final subsequence sub = [1, 2, 3, 10]. The length of sub is our answer.

    It appears the best way to build an increasing subsequence is: for each element num, if num is greater than the largest element in our subsequence, then add it to the subsequence. Otherwise, perform a linear scan through the subsequence starting from the smallest element and replace the first element that is greater than or equal to num with num. This opens the door for elements that are greater than num but less than the element replaced to be included in the sequence.

    One thing to add: this algorithm does not always generate a valid subsequence of the input, but the length of the subsequence will always equal the length of the longest increasing subsequence. For example, with the input [3, 4, 5, 1], at the end we will have sub = [1, 4, 5], which isn't a subsequence, but the length is still correct. The length remains correct because the length only changes when a new element is larger than any element in the subsequence. In that case, the element is appended to the subsequence instead of replacing an existing element.
    '''
    # start with the first element
    sub = [nums[0]]
    # step through the other numbers
    for num in nums[1:]:
        # if the current number is larger than the previous largest, we are increasing
        if num > sub[-1]:
            # therefore, append it to the sequence
            sub.append(num)
        # this number is smaller than the previous largest
        else:
            # Find the first element in sub that is greater than or equal to num
            i = 0
            while num > sub[i]:
                i += 1
            # replace this number
            sub[i] = num
    # NOTE: sub might not be a valid subsequence, since elements get swapped around
    # but, the length of the total array will still be correct
    return len(sub)


    # Approach 3) Improve on approach 2 with binary search
    # instead of doing a linear scan when the new number is smaller than our current largest, we can use binary search to find the index better. 
    '''Algorithm:
        Initialize an array sub which contains the first element of nums.

        Iterate through the input, starting from the second element. For each element num:

            If num is greater than any element in sub, then add num to sub.
            Otherwise, perform a binary search in sub to find the smallest element that is greater than or equal to num. Replace that element with num.

        Return the length of sub.
    '''
    sub = [nums[0]]
    for num in nums[1:]:
        i = bisect_left(sub, num)
        # If num is greater than any element in sub, we can add it as part of LIS
        if i == len(sub):
            sub.append(num)
        # Otherwise, replace the first element in sub greater than or equal to num
        else:
            sub[i] = num
    return len(sub)


def longestCommonSubsequence(text1, text2):
    '''
    Another dynamic programming problem, with a beautiful explanation on leetcode:
    https://leetcode.com/problems/longest-common-subsequence/solution/
        Goes from: greedy -> top-down memoization -> bottom-up tabulation -> dynamic programming

    We can try a greedy approach and see how it falls apart, might not find the optimal solution.
    Then, we can try a top-down recursive approach that compares two possible conditions:
        either the current character is part of the optimal solution
        the current character is not part of the optimal solution

    Since iteration is faster than recursion in many languages, we want to convert it to bottom-up if possible.
    The key insight is seeing that the final answer (comparing both long strings) depends ultimately on the smallest subproblem: comparing their last characters
    Then, we can tabulate (inverse of memoization) the results bottom up, solving the sub-problems until we reach the top.

    Think of the grid picture and "intuitive" solution in the leetcode video.
    When characters match, we add one and check the diagonal (since we have to check the remaining chars in both strings)
    When characters don't match, we have to check between the remainder of both strings.
    There are zeros padded at the edges for the base case: empty strings
        If characters never match, we bubble up these zeros and are left with a zero at the top-left. 
    '''
    
    '''
    First approach with recursive memoization.
    We are checking two cases:
        - Either the first character, paired with its first matching occurence in text2, is part of the solution.
        - Or, it is not part of the solution.
    We need to check both, with special handling for when the first character is not in the string at all.
    If one or both strings are empty, then we return 0 (aka there is no matching subsequence), this is the base case

    Complexities:
        Runtime of O(N^2 * M)
        N * M from checking the position of each string
        Then, another N from searching for a character in a string (text2.find())

        O(N * M) space complexity from storing the answer to each subproblem.
    '''
    @lru_cache(maxsize=None)
    def memo_solve(p1, p2):
        
        # Base case: If either string is now empty, we can't match
        # up anymore characters.
        if p1 == len(text1) or p2 == len(text2):
            return 0
        
        # Option 1: We don't include text1[p1] in the solution.
        # aka the first pair of matching characters is *not* part of the solutions
        option_1 = memo_solve(p1 + 1, p2)
        
        # Option 2: We include text1[p1] in the solution, as long as
        # a match for it in text2 at or after p2 exists.
        first_occurrence = text2.find(text1[p1], p2) # does text1[p1] exist in text2[p2:] ?
        if first_occurrence == -1:
            option_2 = 0
        else:
            # here we get a +1 from characters matching and being part of the solutions
            # the max bubbles up the largest +1 to reach the optimal solutions
            option_2 = 1 + memo_solve(p1 + 1, first_occurrence + 1)
        
        # Return the best option.
        return max(option_1, option_2)
            
    return memo_solve(0, 0)

    '''
    The second approach involves better memoization.
    The condition we are checking for changes.
    Now, if the two first characters are the same, there is no reason not include them!
    But, if they are not the same, then the optimal solution will either be in the remaining chars of text1, or the remaining chars of text2.

    Complexities:
        O(N*M) from checking indexes of both strings.
            No longer searching the other string, just doing an O(1) comparison on the first char.
        Space complexity still N*M from storing all subproblems. 
    '''
    @lru_cache(maxsize=None)
    def memo_solve(p1, p2):
        
        # Base case: If either string is now empty, we can't match
        # up anymore characters.
        if p1 == len(text1) or p2 == len(text2):
            return 0
        
        # Recursive case 1.
        # The first characters match, link them up and check the rest of the string.
        if text1[p1] == text2[p2]:
            return 1 + memo_solve(p1 + 1, p2 + 1)
        
        # Recursive case 2.
        # The first characters don't match, check the rest of the strings.
        else:
            return max(memo_solve(p1, p2 + 1), memo_solve(p1 + 1, p2))
        
    return memo_solve(0, 0)
    

    '''
    Lastly, we can convert this last memozied problem into a DP solution.
    Notice that the longest/largest subproblem we are solving is |text1| * |text2|
    And, that solution depends on solving smaller subproblems.
    In the extreme, the smallest subproblem is simply checking the last characters of each string.
        Consider the base cases:
            - ([], []) -> empty strings, no subsequence
            - ([], *) || (*, []) -> at least one empty, no subsequence
            - ([x], [x]) -> single characters, either they match or they don't
                - If they match, evaluate the rest of the string
                - If they don't match, check both remainders and keep the rest 
    Then, we can build/climb back up and, by the time we reach larger subproblems, the smaller subproblems they depend on will have been solved.

    The easiest way to visualize and implement this is with a 2D grid, where each cell stores the result of the (p1, p2) subproblem.

    NOTE: the visualization in the official solution does an excellent job of showing this, column-wise. 
    Neetcode's solution was row-wise, both are possible
    '''
    # Make a grid of 0's with len(text2) + 1 columns 
    # and len(text1) + 1 rows.
    dp_grid = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
    
    # Iterate up each column, starting from the last one.
    for col in reversed(range(len(text2))):
        for row in reversed(range(len(text1))):
            # If the corresponding characters for this cell are the same...
            if text2[col] == text1[row]:
                dp_grid[row][col] = 1 + dp_grid[row + 1][col + 1]
            # Otherwise they must be different...
            else:
                dp_grid[row][col] = max(dp_grid[row + 1][col], dp_grid[row][col + 1])
    
    # The original problem's answer is in dp_grid[0][0]. Return it.
    return dp_grid[0][0]

    # ## row variation:
    # dp_rows = [[0] * len(text1+1) for _ in range(len(text1)+1)]
    # # step through each row, and col
    # for row in range(len(text1) - 1, -1, -1):
    #     for col in range(len(text2) - 1, -1, -1):
    #         # if the current characters match, add one and check the remaining strings
    #         if text1[row] == text2[col]:
    #             dp_rows[row][col] = 1 + dp_rows[row+1][col+1]
    #         # else, take the max of adjacent cells
    #         else:
    #             dp_rows[row][col] = max(dp_rows[row+1][col], dp_rows[row][col+1])
    # # the answer, if any, is in the top-left entry of the matrix
    # return dp_rows[0][0]

    '''
    Lastly, notice we are only ever computing two columns, or rows, respectively.
    And, once we've computed one set of values, we don't need the "previous" anymore.
    This means we can include one more optimization, where we only keep track of the last two rows/cols.

    We have to make sure we are using the shorter word, to avoid going out of bounds in the comparison
    '''
    # If text1 doesn't reference the shortest string, swap them.
    if len(text2) < len(text1):
        text1, text2 = text2, text1
    
    # The previous and current column starts with all 0's and like 
    # before is 1 more than the length of the first word.
    previous = [0] * (len(text1) + 1)
    current = [0] * (len(text1) + 1)
    
    # Iterate up each column, starting from the last one.
    for col in reversed(range(len(text2))):
        for row in reversed(range(len(text1))):
            if text2[col] == text1[row]:
                current[row] = 1 + previous[row + 1]
            else:
                current[row] = max(previous[row], current[row + 1])
        # The current column becomes the previous one, and vice versa.
        previous, current = current, previous
    
    # The original problem's answer is in previous[0]. Return it.
    return previous[0]


class Solution:
    '''Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.
        Note: You can only move either down or right at any point in time.
    '''
    def minPathSum(self, grid: List[List[int]]) -> int:
        
        # matrix dimensions
        m, n = len(grid), len(grid[0])
        
        memo = {}
        
        def best_path_top_down(row, col):
            
            if (row,col) in memo:
                return memo[(row,col)]
            
            if row == 0 and col == 0:
                memo[(row,col)]  = grid[row][col]
            elif row == 0:
                memo[(row,col)]  = grid[row][col] + best_path_top_down(row, col-1)
            elif col == 0:
                memo[(row,col)]  = grid[row][col] + best_path_top_down(row-1, col)
            else:
                memo[(row,col)]  = grid[row][col] + min(
                    best_path_top_down(row-1, col),
                    best_path_top_down(row, col-1)
                )
                
            return memo[(row,col)] 
            
        return best_path_top_down(m-1, n-1)


        '''Memo solution with array storage'''
                # matrix dimensions
        m, n = len(grid), len(grid[0])
        
        for row in range(m):
            for col in range(n):
                
                if row == 0 and col == 0:
                    continue
                    
                elif row == 0:
                    grid[row][col] = grid[row][col] + grid[row][col-1]
                elif col == 0:
                    grid[row][col] = grid[row][col] + grid[row-1][col]
                else:
                    grid[row][col] = grid[row][col] + min(
                        grid[row-1][col],
                        grid[row][col-1],
                    )
            
        return grid[m-1][n-1]