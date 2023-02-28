
def productExceptSelf(nums):
    '''
    This is trivial if we can use division. Then all we have to take is take the total product, then at each cell divide it by the number at said cell. Sneak: multiply by num ** -1
    Note: zero adds some edge cases when taking cumulative product.

    # build the outputs
    ans = []
    for i in range(len(nums)):
        prod = 1
        for j in range(len(nums)):
            if i == j : continue
            prod *= nums[j]
        ans.append(prod)
    return ans

    When we can't use division, we need two buffers: one for the forward product and another for the backward product. But! That requires more memory. We can use a single buffer creatively as follows:

    - Do a forward cumprod with a padding (prefix) of one.
    - Run a backwards cumprod, also with a padding (postfix) of one and multiply it by the prefix of the previous step.

    The key part is that for each element, we compute the product before/after it, respectively.
    '''

    # output buffer with all ones
    N = len(nums)
    res = [1 for _ in range(N)]

    # run the forward cumprod
    prefix = 1
    for i in range(N):
        res[i] = prefix # note: only set the prefix, don't multiply here
        prefix *= nums[i]

    # run the backward cumprod
    postfix = 1
    for i in range(N-1, -1, -1):
        res[i] *= postfix
        postfix *= nums[i]

    # return the merged-except-self product
    return res


def maxSubArray(nums) -> int:
    '''
    The native approach would be O(n^3).
        for i in range(N):
            for j in range(i, N):
                # init current sum at start (i), check max sum
                for k in range(i+1, j):
                    # manually check if sum increases value, check against global max sum
        There is a way to make it O(n^2) without an auxiliary loop.
            for i in range(N):
                # init sum to current i position, then check max
                for j in range(i+1, N):
                    # add to running sum, check max
    For every element, take every possible sub-array and compute its product. 
        Some saving from caching intermediate products, but overall very ugly. 

    But, this can be done in one pass thanks for a few key details (Kadane's algorithm)
    Namely, the fact that we are simply adding things, and the fact that adding zero does not change a sum.
    Nice Java implementation:
        public static int solve1(int[] input) {

            int sum = input[0];
            int bestSum = sum;

            for (int i = 1; i < input.length; i++) {
                sum = Math.max(input[i], input[i] + sum);
                bestSum = Math.max(sum, bestSum);
            }
            return bestSum;
        }

    Python equivalent:
        # python
        if not nums: return None
        # start off with the best guess
        cur_sum = best_sum = nums[0]
        # check if we can do better
        for i in range(1, len(nums)):
            # take the max between the current number, and adding it to our running sum
            cur_sum = max(nums[i], nums[i] + cur_sum)
            # check if this new sum is better than our running max
            best_sum = max(cur_sum, best_sum)
        # at this point, we will have stored the best max
        return best_sum

    So, we can run through the entire array at once, keeping a running sum.
    Whenever the sum dips negative, we can reset the running sum since we know this number can't possibly be part of the largest total. 
    Then, after adding each number, we check if it's the global max so far.

    Note: we don't have to return which numbers led to the sum, just the value. Makes it a bit easier
    '''
    if not nums: return None

    # start with a random guess, and initialize the current running sum
    maxSub = curSum = nums[0]

    # step through each number
    for n in nums[1:]:

        # if the sum has dipped negative, "reset" it
        curSum = 0 if curSum < 0 else curSum

        # keep accumulating
        curSum += n 
        # check if the current sum is the new max seen so far
        maxSub = max(maxSub, curSum)

    # return the largest sum we found
    return maxSub


    '''Cleaner python using initial values, and checking running sum at each loop.
    No special handling for when sum dips below 0,
    '''
    if not nums: return None
    cur_sum = best_sum = nums[0]
    for i in range(1, len(nums)):
        cur_sum = max(nums[i], nums[i] + cur_sum)
        best_sum = max(cur_sum, best_sum)
    return best_sum


def maxProduct(nums) -> int:
    '''
    This is similar to max sum, with a few key differences:
        - Multiplying by 0 annihilates the current running outputs
        - Multiplying by negatives flips signs (phase shift).
            That means we either tediously keep track of odd vs. even number of negative-multiplies, to know which way the sign goes.
            Or, we find some other clever way of tracking negatives.

    The solution is to keep two running totals, the most negative and most positive seen so far.
        The reason, there could be a very negative large number, that gets multiplied with a current negative running sum to product a giant number
    '''
    # start with random guess
    res = nums[0]
    # keep track of largest positive and negative products
    cur_min, cur_max = 1, 1

    # step through each number
    for n in nums:

        # store this number times the current max, since we double-check in the second min-line
        tmp = n * cur_max

        # check which is larger and negative after multiplying the current number
        # we check the following: (current vs. max), current vs. min, current vs. n
        cur_max = max(n, tmp, n * cur_min)
        cur_min = min(n, tmp, n * cur_min)

        # find the overall new max
        res = max(res, cur_max)

    # return the largest product seen
    return res



def missingNumber(nums):
    '''
    Many ways to solve this, can use Gauss' formula or fancy xor tricks.
        xor is commutative, and eventually two numbers cancel out.
    '''
    # with gauss' formula
    def gauss_sum(n):
        "Sums the first `n` integers together."
        return (n * (n - 1)) / 2
    gsum = gauss_sum(len(nums))
    for n in nums: gsum -= n
    # gsum -= sum(nums) # one-liner subtraction
    return gsum
    # # one-liner everything
    # return (len(nums)*(len(nums)-1)//2) - sum(nums)

    # with xor trick, since xor is its own inverse
    # and xor is commutative / assosciative
    res = len(nums)
    for i, num in enumerate(nums):
        res ^= (i ^ num)
    return res 

    # with difference of two sums
    # key point is we tackle both sums in the same loop, leveraging the index
    # the missing value will "stick out" since it won't have its negative pair
    res = len(nums) # adding the last value which the for-loop which stops at len(n) - 1
    for i in range(len(nums)):
        res += (i - nums[i])
    return res






class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        # # O(n^2) bad
        # max_avg = float('-inf')
        # for i in range(len(nums) - k + 1):
        #     max_avg = max(max_avg, sum(nums[i:i+k]) / k)
        # return max_avg

        # not using the sliding window idea
        # here, set the first sliding window and evaluate the condition/function.
        best_sum = init_sum = sum(nums[i] for i in range(k))

        # slide the window to the rest of the positions
        for i in range(k, len(nums)):
            # intelligently update the window condition/func.
            ## here, we can drop the previous element and add the current one
            init_sum += nums[i] - nums[i-k]
            # if this sum was better, add it
            best_sum = max(best_sum, init_sum)

        # note: we can save division until the end
        return best_sum / k

        # with prefix for practice
        prefix = [0]
        for n in nums:
            prefix.append(prefix[-1] + n)

        max_sum = max(prefix[i+k] - prefix[i] for i in range(len(nums) - k + 1))

        return max_sum / k


class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        # using a hashmap
        seen = {}

        # variable for start of string and the largest value
        max_len = beg = 0

        # step through the whole string
        for j in range(len(s)):

            # add element to the substring
            seen[s[j]] = 1 + seen.get(s[j], 0)

            # we've grown too big, trim from the start until we can add new characters
            while len(seen) > k:
                
                # delete characters from the back
                seen[s[beg]] -= 1
                if seen[s[beg]] == 0:
                    del seen[s[beg]]
                beg += 1 

            # check the current, max string
            max_len = max(max_len, j - beg + 1)

        return max_len


class Solution:
    def findMaxConsecutiveOnes1(self, nums: List[int]) -> int:
        '''Note: simplified version if we're allowed to flip k zeros, with k = 0 

        We can save the window update step, since we can't change anything.
        So, whenever we find a 0, assuming we have a running length so far, we immediately move up
            If there is a longer sequence, it will always be above this one.
        '''
        max_ones = cs = 0

        for i in range(len(nums)):
            if nums[i] == 1:
                cs += 1
            else:
                cs = 0
            max_ones = max(max_ones, cs)

        return max_ones


class Solution:
    def findMaxConsecutiveOnes2(self, nums: List[int]) -> int:
        '''Assuming we can flip at most 2 zeros'''
        # vars for longest sequence, the current left edge of the window
        # and, running number of zeros
        max_ones = beg = nz = 0

        # move right edge of the window along
        for i in range(len(nums)):
            
            # if we've seen a zero, add our count
            if nums[i] == 0:
                nz += 1

            # if the window's invalid, shrink until we fix it
            while nz == 2:
                if nums[beg] == 0:
                    nz -= 1
                beg += 1

            # find running max of longest, fixed window
            max_ones = max(max_ones, i - beg + 1)

        return max_ones

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        
        l = 0
        freq = {}
        maxlen = 0
        for r in range(len(s)):
            # If a character is not in the frequency dict, this inserts it with a value of 1 (get returns 0, then we add 1).
            # If a character is in the dict, we simply add one.
            freq[s[r]] = freq.get(s[r], 0) + 1
             
            # The key point is that we only care about the MAXIMUM of the seen values.
            # Get the length of the current substring, then subtract the MAXIMUM frequency. See if this is <= K for validity.
            cur_len = r - l + 1
            if cur_len - max(freq.values()) <= k:  # if we have replaced <= K letters, record a new maxLen
                maxlen = max(maxlen, cur_len)
            else:                               # if we have replaced > K letters, then it's time to slide the window
                freq[s[l]] -= 1                 # decrement frequency of char at left pointer, then increment pointer
                l += 1
               
        return maxlen


class Solution:
    def longestNiceSubarray(self, nums: List[int]) -> int:
        '''NOTE: think about the numbers aligning, and what it means that *all* values in the array have to meet the condition.'''

        # track the number of nice arrays
        max_nice = 0
        beg = 0

        # NOTE: was missing one variable for running product of the array
        res = 0

        # walk out the window
        for i in range(len(nums)):

            while res & nums[i]:
                res ^= nums[beg] # subtract the number from the start
                beg += 1 # move up the left window

            # check if the running condition remains valid:
            # if we've or'd all the numbers, they should be zero
            res |= nums[i]

            # update the running maximum
            max_nice = max(max_nice, i - beg + 1)

        return max_nice

        '''My wrong solution was close, but missing the idea that *all* values had and to 0'''
        # # track the number of nice arrays
        # max_nice = 1
        # beg = 0

        # # walk out the window
        # for i in range(len(nums)):
            
        #     # if the window has become invalid, shrink it
        #    # NOTE: this might not match the problematic value. only when it's at the front
        #     if nums[i] & nums[beg] != 0:
        #         # no element between beg and i will be zero either, move beg to this spot to check future spots
        #         beg = i

        #     # update the running maximum
        #     max_nice = max(max_nice, i - beg + 1)

        # return max_nice


class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        '''Similar to other windows.

        Had issues with the dynamic size and what this means. 
        Needed a hashmap, that intuition was correct.
        '''
        # track running sum and final max
        max_sum = run_sum = 0 

        # values we've seen
        seen = defaultdict(int)

        # grow the window
        for i in range(len(nums)):
            
            # track current number and increase the running sum
            # "always add to array"
            seen[nums[i]] += 1
            run_sum += nums[i]

            # if our window is large enough, remove the last element
            # if window has become valid/invalid, check for conditions
            # technically, we're moving the window up and handling duplicates
            if i >= k: # need >= because of 0-indexing
                # "drop" the last element from the window
                seen[nums[i-k]] -= 1
                run_sum -= nums[i-k]
                # if there are no more elements, remove it
                if seen[nums[i-k]] == 0:
                    del seen[nums[i-k]]

            # if we've built up to k numbers, check if it's a new max
            # this is checking for whether the current window is valid, it has a single element from each one
            if len(seen) == k:
                max_sum = max(max_sum, run_sum)

        return max_sum



class Solution(object):
    '''
        store the length of previous and current consecutive 1's (separated by the last 0) as pre and curr , respectively.

        Whenever we get a new number, update these two variables accordingly. The consecutive length would be pre + 1 + curr, where the 1 is a zero that got flipped to 1. (note that pre is initialized to -1, meaning that we haven't seen any 0 yet)

    '''
    def findMaxConsecutiveOnes(self, nums):
        # previous and current length of consecutive 1 
        pre, curr, maxlen = -1, 0, 0
        # NOTE: look through base cases, this is beautiful
        for n in nums:
            if n == 0:
                pre, curr = curr, 0
            else:
                curr += 1
            maxlen = max(maxlen, pre + 1 + curr )
        
        return maxlen


class Solution:
    def longestOnes3(self, nums: List[int], k: int) -> int:
        """Generalization of the previous case."""
        # setup initial pointers
        max_len = beg = nz = 0

        for i in range(len(nums)):

            if nums[i] == 0:
                nz += 1

            while nz > k:
                if nums[beg] == 0:
                    nz -= 1
                beg += 1

            max_len = max(max_len, i - beg + 1)

        return max_len



def longestConsecutive(self, nums: List[int]) -> int:
    '''
    HARD: finding the length of the longest continuous sequence in the problem.
    Becomes almost trivially easy if we use an axuiliary set.

    An element is the start of a sequence if it has no immediate left-neighbor.
    If it's the start of a sequence, we can check how many adjacent, consecutive values are in the set.

    Small detail and caution when counting things, vs. when we are indexing (where we don't need to worry about + 1 because of 0-based indexing***)
    '''
    if not nums: return 0
    
    # turn numbers into set
    setNums = set(nums) # O(n) memory
    
    # to start, a single element is its own longest sequence
    longest = 1
    
    # check each number
    for n in nums:
        if (n-1) not in setNums: # means `n` is the start of a sequence
            # starting from n, how many consecutive elements are there?
            tmp = n
            while (tmp+1) in setNums:
                tmp += 1
            # NOTE: need to check with offset of one, since we're counting things
            # starting from I, we make N jumps to reach I+J. But the number of jumps does not count starting point.
            longest = max(longest, tmp - n + 1)
            
    return longest
