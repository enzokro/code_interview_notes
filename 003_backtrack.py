
'''Notes from AlgoMonster:

The example above also introduces two other concepts, backtracking and divide and conquer. The action of retracing steps (e.g. from 2 we first visited 3 depth first and retraced back and visit the other child 4) is called backtracking. Backtracking and DFS are similar concepts and essentially the same thing since in DFS you always "backtrack" after exploring a deeper node. It's like saying I program computers by doing coding. If we really want to make the distinction, then backtracking is the concept of retracing and DFS is the algorithm that implements it. In computer science textbooks [1,2,3], backtracking is often mentioned and associated with combinatorial search problems. We will do the same in the course.

We have two recursive calls dfs(root.left) and dfs(root.right), and return based on results from the recursive calls. This is also a divide and conquer algorithm, i.e. splitting into subproblems of the same type (search in left and right children) until they are simple enough to be solved directly (null nodes or found target) and combine the results from these subproblems (return non-null node). We'll investigate divide and conquer more in a later module.
'''

'''General backtracking pattern:'''
def backtrack(candidate):
    if find_solution(candidate):
        output(candidate)
        return
    
    # iterate all possible candidates.
    for next_candidate in list_of_candidates:
        if is_valid(next_candidate):
            # try this partial candidate solution
            place(next_candidate)
            # given the candidate, explore further.
            backtrack(next_candidate)
            # backtrack
            remove(next_candidate)
    

from typing import List, Optional


'''Visualization of permutations

dfs(nums = [1, 2, 3] , path = [] , result = [] )
|____ dfs(nums = [2, 3] , path = [1] , result = [] )
|      |___dfs(nums = [3] , path = [1, 2] , result = [] )
|      |    |___dfs(nums = [] , path = [1, 2, 3] , result = [[1, 2, 3]] ) # added a new permutation to the result
|      |___dfs(nums = [2] , path = [1, 3] , result = [[1, 2, 3]] )
|           |___dfs(nums = [] , path = [1, 3, 2] , result = [[1, 2, 3], [1, 3, 2]] ) # added a new permutation to the result
|____ dfs(nums = [1, 3] , path = [2] , result = [[1, 2, 3], [1, 3, 2]] )
|      |___dfs(nums = [3] , path = [2, 1] , result = [[1, 2, 3], [1, 3, 2]] )
|      |    |___dfs(nums = [] , path = [2, 1, 3] , result = [[1, 2, 3], [1, 3, 2], [2, 1, 3]] ) # added a new permutation to the result
|      |___dfs(nums = [1] , path = [2, 3] , result = [[1, 2, 3], [1, 3, 2], [2, 1, 3]] )
|           |___dfs(nums = [] , path = [2, 3, 1] , result = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1]] ) # added a new permutation to the result
|____ dfs(nums = [1, 2] , path = [3] , result = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1]] )
       |___dfs(nums = [2] , path = [3, 1] , result = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1]] )
       |    |___dfs(nums = [] , path = [3, 1, 2] , result = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2]] ) # added a new permutation to the result
       |___dfs(nums = [1] , path = [3, 2] , result = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2]] )
            |___dfs(nums = [] , path = [3, 2, 1] , result = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]] ) # added a new permutation to the result
'''


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:

        def dfs(nums, k, path, res):
            if k == 0:
                # # will see the "tree" of valid solutions printed in DFS order
                # print(path)
                res.append(path)

            if len(nums) >= k:
                for i in range(len(nums)):
                    dfs(nums[i+1:], k-1, path + [nums[i]], res)

        res = []
        dfs([i for i in range(1, n+1)], k, [], res)
        return res


        # even cleaner without extra k variable, passing in a single index
        output = []
    
        def backtracking(start, pairs):
            if len(pairs) == k:
                output.append(pairs[:])
                return
            
            # start from the first number, all the way to  n+1
            # in recursive calls, this will slowly move up the list
            # if needed, we could access nums[i] below.
            for i in range(start, n+1):
                backtracking(i+1, pairs + [i])
    
        backtracking(1, [])
        return output

        # a recursive approach, good for reference and to breakdown
        if k == 0:
            return [[]]
        return [pre + [i] for i in range(k, n+1) for pre in self.combine(i-1, k-1)]

        # with reduce instead
        return reduce(
            lambda C, _: [[i]+c for c in C for i in range(1, c[0] if c else n+1)], range(k), [[]])


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:

        def _subsets(nums, path, res):
            res.append(path) # pre-order traversal
            for i,n in enumerate(nums):
                _subsets(nums[i+1:], path + [n], res)
            # res.append(path) # post order traversal

        res = []
        _subsets(nums, [], res)
        return res





class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:

        def dfs(nums, total, path, res):
            # base case: we got to 0
            if total == 0:
                res.append(path)
                return
            # base case, we overshot
            elif total < 0:
                return
            else:
                for i,n in enumerate(nums):

                    # skip over duplicates
                    if (i > 0) and n == nums[i-1]:
                        continue

                    # if this n would overshoot us into negative, can break the loop
                    if n > total:
                        break

                    # try this solution
                    dfs(nums[i+1:], total - n, path + [n], res)

        res = []
        candidates.sort()
        dfs(candidates, target, [], res)
        return res


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def dfs(idx, total, path, res):
            if total == 0:
                res.append(path)
            elif total < 0 or idx >= len(candidates):
                return
            else:
                # need two checks, based on current index
                ## try adding the current number as part of the solution
                if candidates[idx] <= total:
                    dfs(idx, total - candidates[idx], path + [candidates[idx]], res)
                ## try *not* adding the number as part of the solution
                dfs(idx + 1, total, path, res)

        dfs(0, target, [], res)

        return res


class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        def dfs(acc, pos):
            yield acc
            for i in range(pos, len(nums)):
                if i > pos and nums[i] == nums[i-1]:
                    continue
                yield from dfs(acc+[nums[i]], i+1)
        
        return list(dfs([], 0))

    def subsetsWithDup(self, nums):
        nums, result, pos = sorted(nums), [[]], {}
        for n in nums:
            start, l = pos.get(n, 0), len(result)
            result += [r + [n] for r in result[start:]]
            pos[n] = l
        return result

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        # edge case, nothing given
        if not digits: return []

        # map from letters to number
        cmap = {
            '2': [*'abc'],
            '3': [*'def'],
            '4': [*'ghi'],
            '5': [*'jkl'],
            '6': [*'mno'],
            '7': [*'pqrs'],
            '8': [*'tuv'],
            '9': [*'wxyz'],
        }

        # start with all chars from first digit
        res = [[c] for c in cmap[digits[0]]]

        # add chars from all other digits
        for d in digits[1:]:
            res = [o + [c] for o in res for c in cmap[d]]

        res = [''.join(o) for o in res]
        return res

def combinationSum(nums, target):
    '''
    Our first proper example of Backtracking:
        - Incrementally building up the solution, until we hit an invalid state.
        - Then, "back off" and start working back up the chain.

    Think about the picture where we're adding "unique" elements per branch, so we avoid repeats.
    '''
    res = []

    def dfs(i, cur, total):
        # base case 1), we hit a valid target
        if total == target:
            res.append(cur[:]) # make sure to pass a copy
            return

        # base case 2), we are out of bounds or exceeded the target
        if i >= len(nums) or total > target:
            return

        # need to check both branches: repeat the current candidate
        dfs(i, cur + [nums[i]], total + nums[i])
        dfs(i + 1, cur, total) # or, exclude the current candidate and try the next one

    # call it with: the first position/elem, no starting candidates, and an empty total
    dfs(0, [], 0)

    return res
