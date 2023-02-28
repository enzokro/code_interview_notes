def hammingWeight(self, n: int) -> int:
    '''
    The most basic flavor of bit operation.
    There are two operations that shave off the last bit in a number:
        modulo by two
        and-ing with one (this also tells you whether the number is even or odd). 

    The idea is to progressively peek at the last bit, add it to the number, then shift things over. 
    '''
    
    # start with zero number of one bits
    hw = 0
    
    # while the number is not 0...
    while n:
        # add the last bit, nothing happens if it's 0
        hw += n & 1
        # shift the number over
        n >>= 1

    return hw


def countBits(self, n):
    '''
    Many ways to do this via bit tricks and dynamic programming.

    The naive solution is to run hamming weight on each number in a loop.
    This makes it nlogn, which isn't horrible but we can do better.

    The idea comes from seeing that the binary representation "cycles" as the number increase. We get a new significant's place 1, and the previous numbers repeat in the remaining bits.
    So, we store the "base" cases of 0 and 1, and then start tracking the most significant bit.

    When reaching a new significant bit, we know we have a new one to compute.
        Finally, we can index back down to the "beginning" and start running up the cycle, where we've already counted the number of bits in each one. 

    This leads to a nice O(n) solution.
    '''
    ans = [0 for _ in range(n+1)]
    msb = 1
    for i in range(1, n+1):
        if i == msb * 2: # we reached a new significant binary bit
            msb *= 2
        # the new MSB + the number of earlier bits in the cycle
        ans[i] = 1 + ans[i - msb]
    return ans

    # alternative via i & (i-1) trick 
    # ans = [0] * (n + 1)
    # for i in range(1, n + 1):
    #     ans[i] = 1 + ans[i & (i-1)] # this operation removes the least-significant one and zeros out the remaining number
    #     # note: it's a way of "resetting" the cycle to the appropriate place without manually tracking the MSB seen so far
    # return ans

    # '''
    # Love your channel! Here's a slightly simpler solution which I came up with. The idea here is that the number of 1 bits in some num i is: 1 if the last digit of i (i % 1) is 1, plus the number of 1 bits in the other digits of i (ans[i // 2]).
    # NOTE: this is using the LSB as a kind of flag value to "invert" the cycle.
    # '''
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        ans[i] = (i & 1) + ans[i >> 1]
    return ans


'''Adding binary numbers:
Consider edge cases:
    0 + 0
    1 + 1
    1 | 0

Then, what happens with the carry value
For the first two, we need special considerations:"
    did we consume it, or not?
    Do we need to set it? Keep it?

Lastly, if we still have a carry at the end, we need consume it
'''
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        
        # pad the string if necessary
        na, nb = len(a), len(b)
        if na < nb:
            nz = ''.join('0' for _ in range(nb - na))
            a = f'{nz}{a}'
            max_n = nb
        elif na > nb:
            nz = ''.join('0' for _ in range(na - nb))
            b = f'{nz}{b}'
            max_n = na
        else:
            max_n = na
        # # cleaner version
        # n = max(len(a), len(b))
        # a, b = a.zfill(n), b.zfill(n)
        
        # store overall response and carry bit
        res = []
        carry = False
        
        # work backward from the strings
        for i in range(max_n - 1, -1, -1):
            
            # case 1) Both zero digits
            if a[i] == '0' and b[i] == '0':
                # if we have a carry, add it in and remove carry flag
                if carry:
                    val = '1'
                    carry = False
                # else, these sum to 0
                else:
                    val = '0'
            
            # case 2) both ones
            elif a[i] == '1' and b[i] == '1':
                # if we have a carry, it's still fixed to one and we keep the carry
                if carry:
                    val = '1'
                # else, we add the zero and set the carry flag
                else:
                    val = '0'
                    carry = True

            else: # either one is 1/0
                val = '0' if carry else '1'
            
            res.append(val)
                    
        # add surviving carry if needed
        if carry:
            res.append('1')

        # because we started from the back, we need to reverse our answer
        return ''.join(res[::-1])


        '''Crazy way to do it via bit manipulation'''
        x, y = int(a, 2), int(b, 2)
        while y:
            answer = x ^ y
            carry = (x & y) << 1
            x, y = answer, carry
        return bin(x)[2:]


# below with clever trick to reverse bits in byte with 3 ops
import functools
class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        ret, power = 0, 24
        while n:
            ret += self.reverseByte(n & 0xff) << power
            n >>= 8
            power -= 8
        return ret

    # memoization with decorator
    @functools.lru_cache(maxsize=256)
    def reverseByte(self, byte):
        return (byte * 0x0202020202 & 0x010884422010) % 1023

