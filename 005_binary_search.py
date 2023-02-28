def findMinRotatedSortedArray(nums):
    '''
    This is a very tricky one, lots of edge cases.
    Basically, we have an increasing sorted array that has been shifted.

    That means we have two "zones" of increasing values.

    Note that neetcode is missing a few edge cases, seems checking for the right values is "cleaner"
    '''
    res = nums[0]
    beg, end = 0, len(nums) - 1

    while beg <= end: # to removed saved res: beg < end
        if nums[beg] < nums[end]:
            res = min(res, nums[beg])
            break

        m = beg + (end-beg)//2
        res = min(res, nums[m])

        # NOTE: this condition seems key re handling the pivot
        if nums[m] >= nums[beg]: # to remove saved res: nums[m] > nums[end]
            beg = m + 1
        else:
            end = m - 1 # to remove saved res: end = m

    return res # after removing saved res: nums[beg]

#    #look at drawing in notebook. can see how r = m and the specific condition handles the pivot, and leaves l at the true min
#    def findMinClean(self, nums: List[int]) -> int:
#         l, r = 0, len(nums) - 1
#         assert r >= l, "array is empty"
#         # can search exclusive. If we have two values, then the right cannot be the min
#         while l < r:
#             m = l + (r - l) // 2
#             if nums[m] > nums[r]:
#                 l = m + 1
#             else:
#                 r = m
#         return nums[l]


def searchRotatedArray(nums, target) -> int:     
    '''
    This is the step from up from finding the minimum or a rotated, sorted array.

    The basic idea is the same: we have to figure out whether we are in the left or right sorted portions of the array.

    But from there, we need a double-condition check to know whether the target is above or below the midpoint, and whether it is greater/less than the beginning and end pointers, respectively. 

    It is also a <= condition here, need to investigate more. 
    '''   
    beg = 0
    end = len(nums) - 1

    # search inclusive, our answer could be the last element
    while beg <= end:
        mid = beg + (end-beg) // 2
        
        # if we found the target, return it
        if nums[mid] == target:
            return mid
        
        # left sorted portion
        elif nums[mid] > nums[end]:
            # the target is in the left-most portion, we don't have to search any of the right
            if nums[beg] <= target <= nums[mid]:
                end = mid - 1
            # the target is either larger than the middle, or smaller and wrapped around.
            # in either case, we have to search the right-most portion
            else:
                beg = mid + 1
                
        # right sorted portion
        else:
            # the flip cases of the above, except now we check the end value.
            # note: it's easier to compare if the target is between the middle and bounds, respectively
            if nums[mid] <= target <= nums[end]:
                beg = mid + 1
            else:
                end = mid - 1
    # did not find the target, return -1       
    return -1



class Solution:
    def maxLength(self, ribbons: List[int], k: int) -> int:
        
        total = sum(ribbons)
        if total < k:
            return 0
        M = max(ribbons)
        
        # binary search, O(nlogn)/O(1)
        def check(x):
            # O(n) return True if we can get k ribbons of length = x
            res = 0
            for rib in ribbons:
                res += rib // x
                if res >= k:
                    return True
            return False
        
        # with a couple of optimizations to the indexes
        l, r = max(1, M // k), min(M, total // k)
        while l <= r:
            m = (l + r) // 2
            if check(m):
                # we can get k ribbons of length m, should increase
                l = m + 1
            else:
                r = m - 1
        return r
        
        
#         # minPossible = 1             # minimum possible ribbon length
#         # maxPossible = max(ribbons)  # maximum possible ribbon length
        
#         total = sum(ribbons)
#         if k > total:
#             return 0
#         M = max(ribbons)
        
#         # with a couple of optimizations to the indexes
#         minPossible, maxPossible = max(1, M // k), min(M, total // k)
        
#         while minPossible <= maxPossible:
            
#             currentLength = minPossible + (maxPossible-minPossible) // 2 # length of current try
#             numOfPiecesWithcurrentLength = 0

#             for ribbon in ribbons:  # getting number of possible ribbons with current length
#                 numOfPiecesWithcurrentLength += ribbon//currentLength 
                
#                 # early stopping
#                 if numOfPiecesWithcurrentLength >= k:
#                     minPossible = currentLength + 1
#                     break
                    
#             if minPossible != currentLength + 1:
#                 maxPossible = currentLength - 1

#         return maxPossible


'''Finding the first bad software version.

Parallels to finding the minimum element in a sorted array
Or, more specifically, in finding a certain value.

There are four general cases:
    - Cur and next values are good:
        Bad happens later in array
    - Cur and previous are bad:
        Bad happened earlier in the array
    - Cur was good, next is bad
        We found the bad value, next is bad
    - Cur was bad, prev was good
        We found the bad value, it's the current one

Aside from this, we need to check edge cases, at the end of the array and start
'''
class Solution:
    def firstBadVersion(self, n: int) -> int:
        # try:
        #     return next(i+1 for i in range(n-1, -1, -1) if not isBadVersion(i))
        # except StopIteration:
        #     return
        
        # try binary search, we are trying to find the pivot
        l, r = 1, n
        
        while l <= r:
            
            mid = l + (r - l)//2 # ((r + l) // 2)
            
            # find status of current position
            is_bad = isBadVersion(mid)
            
            # check the values
            if is_bad:

                # edge cases at the beginning and end of the array
                if (mid + 1 > n) or (mid - 1 < 1):
                    return mid

                # if the previous value is also bad, we need to search the left portion of the array
                if isBadVersion(mid-1):
                    r = mid
                # else, we found the pivot, version mid+ is bad
                else:
                    return mid
            else:
                # if the current value wasn't bad, but the next one was, we found the pivot
                if isBadVersion(mid+1):
                    return mid + 1
                # else, both versions were good, the pivot happens later
                else:
                    l = mid


        # ## much simpler version
        # # try binary search, we are trying to find the pivot
        # l, r = 1, n
        
        # while l < r:
        #     mid = l + (r - l) // 2
        #     if isBadVersion(mid):
        #         r = mid
        #     else:
        #         l = mid + 1
        # return l
            


def reverseBits(n):
    '''
    The idea here is we need to peel off the last bit, then drop it to the very edge. 

    We have some help/guarantees because the numbers are 32 bits

    The main idea:
        pop off the current bit (via & 1)
        drop it into correct spot in the output (via |)
        right-shift the number down to prepare the next bit
    '''
    res = 0
    for i in range(31, -1, -1):
        # grab and shift left-most bit
        cur_bit = n & 1
        cur_bit_shifted = cur_bit << i
        # drop it into the correct slow
        res = res | cur_bit_shifted
        # one-liner version
        #res |= (n & 1) << i
        # shift the current number to prepare the next bit
        n >>= 1
    return res