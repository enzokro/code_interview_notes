def maxProfit(prices) -> int:
    '''
    MaxProfit

    Naive soultion is checking every value against every-other. Becomes O(n^2). A better solution invovles two pointers, keeping track of the lowest price seen so far.

    The more general operations:
     - We have a time series
     - There is a relationship between those values
     - We want the two values that maximize one aspect (or a function on) those relationship(s).
    '''
    # error checking, when prices are empty or we're given a single day
    # if not prices or len(prices) == 1:
    if len(prices) <= 1:
        return 0

    # make the two pointers to step through days
    # right pointer always steps through days, left pointer tracks the lowest price seen so far
    buy, sell = 0, 1

    # max profit seen so far 
    maxP = 0

    # step through all days
    while sell < len(prices):
        # check if beginning price is lower than current price (i.e. a profit if we sold here)
        if prices[buy] < prices[sell]:
            # compute the profit and check if it's the max
            profit = prices[sell] - prices[buy]
            maxP = max(maxP, profit)
        else:
            # the current price we would sell at is actually a new low (or the same) aka a new potential for larger profit
            # take this lower price day as the new potential buy date
            buy = sell

        # always move to the next day
        # in the case we found a new lower price, this bumps us to the next one
        sell += 1

    # return the max profit we saw
    return maxP


def twoSum(nums, target) -> int:
    '''
    The main idea: given an iterable (nums), and a function (+), do we have the elements to reach a desired outcome (target)? In this case the iterable is a group of numbers, the function is summation of two numbers, and the outcome is a certain given number. 

    The naive solution is to manually check each number against the other, and stop if we find the target.
    The early stopping once the target is found helps, but the solution is still O(n^2).

    Instead, we can take a clever approach: take the difference (inverse?) of the current element and the target. Then, store the index of this number in a hashmap. If we then find another number whose difference is already in the map, that means we found a match! Example:

    Assume the target is c, and we have two numbers a and b where a + b = c
    We step through the numbers, reaching a:
        diff_a = target - a (== b )
        Then, we store the number in the map:
        prevMap[a] = i
    We keep stepping through the numbers, and eventually we reach b:
        diff_b = target - b (== a)
        But! We already have a in the map from before, with its index.

    That means we can return the index of the current number and the previously stored one.
    The two of them together will sum to the target
    '''
    residualMap = {}

    for i,num in enumerate(nums):
        # get current number and find the delta
        diff = target - num
        # if the delta exists in the map, we found a matching target, return
        if diff in residualMap:
            return [i, residualMap[diff]]
        residualMap[num] = i
    # if we did not find anything, target cannot be reached. 
    # return  # can be implicit

class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        nums.sort()
        answer = -1
        left = 0
        right = len(nums) -1
        clip = k / 2
        while left < right:
            sum = nums[left] + nums[right]
            if (sum < k):
                answer = max(answer, sum)
                left += 1
            else:
                right -= 1
            # small optimization, break when we shoot above half of the array
            if nums[left] > clip:
                break
        return answer

    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        # binary search
        answer = -1
        nums.sort()
        for i in range(len(nums)):
            j = bisect_left(nums, k - nums[i], i + 1) - 1
            if j > i: # if the insertion point is above, grab the element right before
                answer = max(answer, nums[i] + nums[j])
        return answer


def threeSum(nums):
    '''
    Next level of the famous two sum problem. In this case we want three numbers that sum to zero. 

    Note: if you have a given number `a`, then it becomes a variant of two sum:
        We need two other numbers, b and c, that together sum to negative a.

    We also need some special handling for repeated values, and making sure that solutions are unique. 
    
    Naive approach is O(n^3) checking each triplet of values

    The main good idea via binary search:
        - First, sort the array, so we have some guarantee about ordering and size
        - For each number, go ahead and skip duplicates.
        - Then, take this current number as `a`
        - Take b and c as the number right after, and the one at the far right.
        - Check the sum of the three numbers:
            - If above zero, the sum is too big, drop down the larger number.
            - If the sum is below zero, the sum is too small, move up the number.
            - If we find a value that is zero, append it to the results.
            - NOTE: here is the tricky part. Move up the left pointer, so to to the next number.
                - Then, keep skipping the same number while it exists and we are in bounds.
                - This prevent the duplicate triplets. 
    '''
    res = [] # note: could use a hash set to avoid checks for duplicates

    # sort the numbers: n * log(n)
    nums.sort()

    for i,a in enumerate(nums):
        # skip duplicates after the first element
        if i and nums[i-1] == a:
            continue # after this loop, nums[i] will be a non-repeated number

        # set two pointers: to the next number, and to the end of the array
        beg, end = i+1, len(nums) - 1

        # continue while the binary search if valid:
        # exclusive is fine, we don't want start==end since we need unique triplets
        while beg < end:

            # current candidate for a sum to 0
            # like two-sum with target == -a
            cur = a + nums[beg] + nums[end]

            # if the sum was too big, try a smaller number
            if cur > 0:
                end -= 1
            # if the sum was too small, try a larger number
            elif cur < 0:
                beg += 1
            else: # found a target triplet that sums to 0
                res.append([a, nums[beg], nums[end]])

                # skip over duplicates
                while beg < end and nums[beg] == nums[beg + 1]: #6
                    left += 1
                while beg < end and nums[end] == nums[end-1]:#7
                    right -= 1    #8

                # whether or not there were duplicates, we have to move in the pointers
                beg += 1
                end -= 1 # can safely bring down the right pointer as well, same us one comparison

    return res # set(res)

    '''Slight variant with two-sum
    '''
    n = len(nums)
    if n < 3:
        return []
    
    res = []
    nums.sort()
        
    for i, v in enumerate(nums[:-2]):
        if i != 0 and v == nums[i-1]:
            continue
        
        l = i+1
        r = n-1
        target = -v
        
        while l < r:
            if nums[l] + nums[r] == target:
                res.add((v, nums[l], nums[r]))

                # skip over duplicates
                while l < r and nums[l] == nums[l + 1]: #6
                    l += 1
                while l < r and nums[r] == nums[r - 1]:#7
                    r -= 1    #8

                l += 1
                r -= 1
            elif nums[l] + nums[r] > target:
                r -= 1
            else:
                l += 1
    
    return res


class Solution:
    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        if len(nums) <= 2:
            return 0
        
        nums.sort()
        num_triplets = 0
        
        for i,a in enumerate(nums):
            
            beg, end = i + 1, len(nums) - 1
            
            while beg < end:
                cur = a + nums[beg] + nums[end]
                if cur < target:
                    # HERE WAS THE RUB
                    # MOTIVATED BY FACT WE DON'T NEED UNIQUE VALUES, OR CARE ABOUT DUPS
                    ## ALL TRIPLETS ARE VALID
                    # if (i,j,k) works, then (i,j,k), (i,j,k-1),..., 
                    # (i,j,j+1) all work, totally (k-j) triplets
                    '''NOTE: this is also the main difference with 3Sum-smallest
                        In that case, we care about the specific one of these triplets that has the smallest difference.
                        In other words, we have to search among the valid triplets listed above.

                        Here, we simply need to count all such triplets.

                        And, for 3Sum-smaller/smallest, we can't use the hasmap approach because we don't have a specific, target value to lookup.
                    '''
                    num_triplets += end - beg
                    beg += 1
                else:
                    end -= 1
                    
        return num_triplets


class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        
        nums.sort()
        min_diff = float('inf')
        
        for i,a in enumerate(nums):
            # small check to skip duplicate values
            if i > 0 and nums[i-1] == nums[i]:
                continue
                
            # scan the rest of the array
            beg, end = i+1, len(nums) - 1
            
            while beg < end:
                cur = a + nums[beg] + nums[end]
                
                # if we happen to find the target exactly
                if cur == target:
                    return target
                
                # else, we're either above or below
                # the absolute here saves from tracking another variable based on sign
                if abs(target - cur) < abs(min_diff):
                    min_diff = target - cur
                
                # move the pointers along
                if cur > target:
                    # if our answer is bigger, try a smaller number
                    end -= 1
                else:
                    # if our answer is smaller, try a bigger number
                    beg += 1
                # in both cases, we're trying to better zone in
                    
        # in the case that min_diff is negative, we add
        # in the case min_diff is positive, we subtract
        return target - min_diff


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        
        nums.sort()
        res = set()
        
        for i,a in enumerate(nums):  
            for j in range(i+1, len(nums)):
                
                # find the complement we are searching for
                tar = target - nums[i] - nums[j]
                
                # setup the pointers for the rest of the list
                c = j + 1
                d = len(nums) - 1
                
                # while these don't cross
                while c < d:
                    
                    # compute the complement
                    complement = nums[c] + nums[d]
                    
                    # if we've seen this value before
                    if complement == tar:
                        
                        # add if it's a valid candidate
                        candidate = tuple(nums[o] for o in [i, j, c, d])
                        if candidate not in res:
                            res.add(candidate)
                        c += 1
                        d -= 1
                    
                    # if our complement overshoots the target, we need to try a smaller value
                    elif complement > tar:
                        d -= 1
                    # else our complement undershot the target and we need a larger value
                    else:
                        c += 1

        return list(res)


        # slight alternative, generalized to k-sum
        # main idea is to recursively break down the problem until you're solving 2-sum with a running: target - num[0] - num[1] - num[2] ... - nums[k-2]
        def dfs(l, r, k, target, path, out):  # [l, r] inclusive

            # if we're reached the two-sum problem for target
            if k == 2:

                # classic 2sum
                while l < r:
                    # we've found the complement
                    if nums[l] + nums[r] == target:
                        # found a candidate, add it to our result
                        out.append(path + [nums[l], nums[r]])

                        # skip over duplicates from both sides
                        while l < r and nums[l] == nums[l+1]: l += 1  # Skip duplicate nums[l]
                        while l < r and nums[r] == nums[r-1]: r -= 1  # Skip duplicate nums[r]
                        # while l < r and nums[l] == nums[l+1]:
                        #     l += 1  # Skip duplicate nums[l]
                        #     if nums[r] == nums[r-1]:
                        #         r -= 1  # Skip duplicate nums[r]


                        # increment the pointers
                        l, r = l + 1, r - 1

                    elif nums[l] + nums[r] > target:
                        r -= 1  # Decrease sum
                    else:
                        l += 1  # Increase sum

                # base case for 2sum, return
                return

            # else, we still need to breakdown the problem further to make it 2sum
            # in other works, at this point we still have some 3+ k-sum problem.
            while l < r:
                '''What we are doing:
                    - (l + 1): move up to the next number
                    - r: we are still at the right edge of the array
                    - (k - 1): we're going to add in a new number to our target - sum(nums[...])
                    - (target - nums[l]): the updated target, that we will search for the complement
                    - (path + nums[l]): add current number to path as a candidate solutions
                    - out: the running output array with all solutions so far
                '''
                dfs(l + 1, r, k - 1, target - nums[l], path + [nums[l]], out)

                # skip duplicate nums[l], important to avoid needless recursive calls
                while l < r and nums[l] == nums[l+1]:
                    l += 1
                # but! we still need to keep the same right edge. can only skip duplicates from the right edge once we're in 2sum

                # move up the left pointer to keep calling dfs
                l += 1


        def kSum(k):  # k >= 2
            ans = []
            nums.sort()
            '''Arguments:
             - 0 : the starting left pointer
             - len(nums) - 1: the starting right pointer
             - k : the generalized k for the solution
             - target: the running target we will subtract from to find a complement
             - []: the current path, will start empty. very similar to perms/combs
             - ans: the answer array where valid k-sums are added
             '''
            dfs(0, len(nums)-1, k, target, [], ans)
            return ans

        return kSum(4)
             

class Solution:
    def fourSumCount(self, *nums) -> int:
        
#         cnt = 0
#         res = {}
        
#         for a in nums1:
#             for b in nums2:
#                 res[a + b] = 1 + res.get(a+b, 0)
                
#         for c in nums3:
#             for d in nums4:
#                 cnt += res.get(-(c+d), 0)
                
#         return cnt


    # def product(*args, repeat=1):
    #     # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    #     # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    #     pools = [tuple(pool) for pool in args] * repeat
    #     result = [[]]
    #     for pool in pools:
    #         result = [x+[y] for x in result for y in pool]
    #     for prod in result:
    #         yield tuple(prod)
        
        import itertools
        def get_k_counts(first_arrs, second_arrs):
            
            # find the total sum
            res = {}
            cnt = 0
            
            for elems in itertools.product(*first_arrs):
                _sum = sum(elems)
                res[_sum] = 1 + res.get(_sum, 0)
                
            for elems in itertools.product(*second_arrs):
                _sum = -sum(elems)
                cnt += res.get(_sum, 0)
                
            return cnt
        
        split_idx = len(nums)
        cnt = get_k_counts(nums[:split_idx//2], nums[split_idx//2:])
        return cnt
                



def maxArea(self, height: List[int]) -> int:
    '''
    Computing the largest container possible.

    Naive approach is an O(n2) double loop that checks each combination.
    But, this gets expensive.

    There is an O(n) solution that involves starting at left/right edges.
    The reason being, we have a linear formula for the area: height * width
    If we start at the two ends of the array, we are setting up to maximize width.
        Have two variables to "optimize" for, and we choose to control one of them.
    And, we unfortunately have no control over height. 

    Then, we can drop the smaller of the two bars and bring the search up/down to try and find a taller one. 
    '''
    # max area so far (negative area makes no sense)
    maxW = 0
    
    # start pointers at left/right of array
    beg, end = 0, len(height) - 1
    
    # process the entire array
    # ok to go exclusive: we need the two pillars to be different
    while beg < end:
        
        # compute current area
        W = (end - beg) * min(height[beg], height[end])
        # check if this area is a new maximum
        maxW = max(maxW, W) # (maxW, W)[W > maxW]
        
        # check and drop the shorter bar of the two so far
        if height[beg] > height[end]:
            end -= 1
        else:
            beg += 1
            
    return maxW

def lengthOfLongestSubstring(self, s: str) -> int:
    '''Two-pointer idea, moving up the left one based on set membership'''
    
    # set to count non-repeating characters
    sub = set()
    # left pointer at start at string
    l = 0
    # longest non-repeating substring
    res = 0
    
    # sweep right pointer along string
    for r in range(len(s)):
        # while the right pointer points to a duplicate...
        while s[r] in sub:
            # remove the duplicate and move up the left pointer
            # this may have to run several times, until we hit the character that's been repeated...
            # but, because we wan't continuous substring, this couldn't have been part of the solution anyway
            sub.remove(s[l])
            l += 1
            
        # at this point there are no duplicates
        # should always auto-get here on the first iteration
        # so, add this character to the subtring
        sub.add(s[r])
        
        # two ways to update the max: length of the hashset so far, or with pointer math
        # res = max(res, len(sub))
        res = max(res, r - l + 1) # plus one because we are counting elements, not indexes or hops
        
    return res
    

def characterReplacement(self, s: str, k: int) -> int:

    '''Solve with two pointers
    
    Main idea is to have a left pointer, and a right one that sweeps through the whole window.

    Then, keep a running update of our character frequency map.

    While the window is invalid, aka we'd need more edits than we have available,
        Move up the left pointer and pop a count off the char map
    After this, the current window is implicitly valid.

    At that point, we can check if the new valid, edited window is the longest we've seen so far.

    A small but important optimization is keeping a single max of the characters so far 'maxf'. Because of the window logic, we never have to change this value. 
    '''
    if not s: return ''
    

    count = {} # char map for occurences
    longest = 0 # longest substring we've seen so far

    # start left pointer at the beginning
    l = 0

    # NOTE: optimization with maxfreq
    maxf = 0

    # move left pointer throughout the entire string
    for r in range(len(s)):

        # update character map
        count[s[r]] = 1 + count.get(s[r], 0) # nice way of handling if/else
        maxf = max(maxf, count[s[r]])

        # while the window is invalid, move up the left pointer
        # NOTE: this is the key
        # while (r - l + 1) - max(count.values()) > k:
        while (r - l + 1) - maxf > k:
            count[s[l]] -= 1
            l += 1

        # else, the window is implicitly invalid and we check the new, substituted length against our old one
        # find the longest string so far
        longest = max(longest, r - l + 1)

    return longest



def minWindow(self, s: str, t: str) -> str:
    '''Find the minimum substring that contains all the chars in t.

    Another two pointer (hard) problem.
    Basically, we keep two maps and counts that track membership of valid, needed characters.

    There seems to be a pattern:
        While (the condition is invalid):
            Take action to make the condition valid
        Then, the condition is implicitly true
            And we can update variables

    Think of neetcode's diagram with long string and BANC, and how we move the right pointer and slide up the left to both:
        Find valid chars by growing the window
        Move up the left pointer (shrink the window) to try and build a shorter solution.
    '''
    
    # base case: empty target t
    if t is '': return ''
    
    # initialize two hashmap / windows: one for the chars we've seen so far
    # and another for the total characters we need inside of t
    need_t, have_s = {}, {}
    
    # initialize the need_t window, this won't change and is our success criteria
    for c in t:
        need_t[c] = 1 + need_t.get(c, 0) # nice way of initializing dict
        
    # now, initialize our two count variables
    # this flags when we've found enough characters in the window to match t
    have, need = 0, len(need_t)
    
    # to store our results: beg and end of the string, and its length
    res_beg, res_end = -1, -1
    res_len = len(s) + 1 
    
    # start our left pointer, and slide the right one through
    l = 0 
    for r in range(len(s)):
        
        # grab the current character and add it to the window we have so faR
        c = s[r]
        have_s[c] = 1 + have_s.get(c, 0)
        
        # if this character is what we needed for our solution
        if have_s[c] == need_t.get(c):
            have += 1
            
        # if we found all the chracters we need, try decreasing the window to keep the characters valid
        while have == need:
            
            # update the result if it's shorter than our previous best
            if (r - l + 1) < res_len: # this should always run at least once
                res_beg = l
                res_end = r
                res_len = r - l + 1
            
            # try removing chracters, from the left, to see if we can get shorter
            have_s[s[l]] -= 1
            if s[l] in need_t and have_s[s[l]] < need_t[s[l]]:
                have -= 1 # if we dropped a target char, we might need to gather more
                
            # move up the left pointer
            l += 1
            
    return s[res_beg:res_end + 1] if res_len != len(s) + 1 else '' 
            
            

class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        '''
        Variant of merging two sorted arrays.
        Except in this case, one half of the array will be decreasing, the other increasing.
        Thanks to squaring, a negative number might be "larger" than the current, positive square
            That's the situation we are handling for and fixing
        '''
        
        # square the array
        res = [0] * len(nums)
            
        # two pointers through array
        beg = 0
        end = write = len(nums) - 1
        
        for write in range(len(nums) -1, -1, -1):
            
            if abs(nums[beg]) > abs(nums[end]):
                pos = beg
                beg += 1
            else:
                pos = end
                end -= 1
                
            res[write] = nums[pos] ** 2
            
        return res
    
        
        

