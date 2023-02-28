def cyclic_sort(nums):
    '''
    Note: this only works when we have the exact range of numbers.
    For example, N = 7, and we have the values 0...6

    This is why it works for leetcode's missing number.

    But, in the case we have different or spread values, we need a double loop.
        To find the correct index location.
    '''
    start = 0
    while start < len(nums):
        num = nums[start]
        if num < len(nums) and num != start:
            nums[start], nums[num] = nums[num], nums[start]
        else:
            start += 1
    return nums

def get_a():
    a = [6, 2, 3, 4, 1, 0, 5]
    return a

a = get_a()

cyclic_sort(a)


class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        
        # naive solution with nlogn time, n space
        res = set()
        nums.sort()
        for i in range(len(nums) - 1):
            if nums[i] == nums[i+1]:
                if nums[i] not in res:
                    res.add(nums[i])
        return list(res)

        # attempting cyclic sort
        for i in range(len(nums)):
            # cyclic sort:
            ## while the number at index i does not match the value of nums[i] (subtracting one for 0-indexed),
            ## try swapping the values. In the case the number is in the correct spot, we don't do anything
            ## that means, if we've already put a number into its slot, we will "skip" another instance of that same number
            ## then we can simply return all the values that are not in their correct spot. 
            while not(nums[i] == nums[nums[i] - 1]):
                # NOTE: the tuple ordering matters!
                # "first", we have to put the value at [i] into the current slow
                # then, we can move the value at [nums[i] - 1] down into the ith slot
                # this saves us from the tmp variable:
                # tmp = nums[nums[i] - 1]
                # nums[nums[i] - 1] = nums[i]
                # nums[i] = tmp
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]

        # note: leveraging "enumerate"'s API to save ourselves a few additions
        return [n for i,n in enumerate(nums, 1) if i != n]    


        ## lastly, can use the array itself a hasmap to flag when values are in their correct spot, or not
        ans = []
        for num in nums:
            if nums[abs(num)-1] < 0:
                # only if a number has been seen already will it trip this condition
                # else, it doesn't mean that *all* repeated numbers will be negative, or that we can look at positives only
                ans.append(abs(num))
            else:
                nums[abs(num)-1] *= -1
        return ans


class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:

        # with cyclic sort
        # # cyclic sort
        # for i in range(len(nums)):
        #     while not(nums[i] == nums[nums[i]-1]):
        #         nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        # # add all numbers where the index is not the value
        # missing = []
        # for i,n in enumerate(nums, 1):
        #     if i != n:
        #         missing.append(i)
        # return missing

        # Iterate over each of the elements in the original array
        for i in range(len(nums)):
            
            # Treat the value as the new index
            new_index = abs(nums[i]) - 1
            
            # Check the magnitude of value at this new index
            # If the magnitude is positive, make it negative 
            # thus indicating that the number nums[i] has 
            # appeared or has been visited.
            if nums[new_index] > 0:
                nums[new_index] *= -1
        
        # Response array that would contain the missing numbers
        result = []    
        
        # Iterate over the numbers from 1 to N and add all those
        # that have positive magnitude in the array 
        # for i in range(1, len(nums) + 1):
        for i,n in enumerate(nums, 1):
            if n > 0:
                result.append(i)
                
        return result 


class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        # beautiful solutions using boolean/None flag to mark seen numbers
        # dup will go None -> True -> False on the duplicate
        # the unseen value will remain None
        slots = [True] + [None] * len(nums)
        for n in nums:
            slots[n] = not slots[n]
        return [slots.index(False), slots.index(None)]
        
        # using Gauss' sum formula to find the supposed sum of ints
        # then, using a set to find the missing number
        # then (sum(nums) + missing will be the true sum, but we will have one extra dup)
        # so, we subtract Gauss' sum from this True+dup, and will be left with dup
        n = len(nums)
        s = n * (n + 1) // 2
        miss = s - sum(set(nums))
        duplicate = sum(nums) + miss - s
        return [duplicate, miss]
        

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        '''All nums repeated except one, find it.'''

        # use the fact that xor is its own inverse
        res = nums[0]
        for n in nums[1:]:
            # every repeated number will "cancel" out, only the lone number will survive
            res ^= n
        return res


class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        '''Given n numbers in range 1,n+1, one has been repeated and the other is missing.
        
        Find them both
        '''

        # # with sorting
        # nums.sort()
        # for i in range(1, len(nums)):
        #     if nums[i] == nums[i-1]:
        #         return nums[i]

        # # with hashset
        # seen = set()
        # for n in nums:
        #     if n in seen:
        #         return n
        #     else:
        #         seen.add(n)

        # # with hasmap
        # seen = {}
        # for n in nums:
        #     seen[n] = 1 + seen.get(n, 0)
        # for n,cnt in seen.items():
        #     if cnt == 2:
        #         return n

        # in-place negative markings
        dup = None
        for i,n in enumerate(nums):
            v = abs(n)
            if nums[v] < 0:
                dup = v
            else:
                nums[v] *= -1
        # restore the array
        # NOTE: can drop if speed is most important
        for i in range(len(nums)):
            nums[i] = abs(nums[i])
        return dup

