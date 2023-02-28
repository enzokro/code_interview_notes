
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    '''Easily done via a hashmap.
    
    The main idea is to count all the characters in each string.
    Then, group them by counts in a hashmap
    
    At the end, all anagrams will be guaranteed grouped since their counts match
    '''
    
    res = defaultdict(list)
    
    for s in strs:
        # note: can't be a map because we lose values
        counts = [0] * 26
        
        # add up the char counts
        for c in s:
            counts[ord(c) - ord('a')] += 1
            
        # add it to result
        res[tuple(counts)].append(s)
        
    return res.values()

def isPalindrome(self, s: str) -> bool:
    '''After some processing, cleaning up the string...

    Setup two pointers, left and right
        Check if the current characters are the same
        If they are, increment/decrement the pointers
        If they are not, this is not a palindrome, return False
    '''

    
    # get lower case version of the cell without spaces
    # NOTE: isalnum() to check alphanumeric characters
    s = ''.join([o for o in s if o.isalnum()]).lower()
    
    # base case: single character or empty string
    if len(s) <= 1: return True
    
    # set up left and right pointers
    l, r = 0, len(s) - 1
    
    # while pointers don't cross
    while l < r:
        # if characters are the same, advance poitners to check the next one
        if s[l] == s[r]:
            l += 1
            r -= 1
        # not a palindrome, return false
        else:
            return False

    # all characters matched, return True
    return True


    '''Without python tricks to skip over spaces'''
    # base cases: single character and empty string are technically palindromes
    if len(s) <= 1: return True

    # # python trick to get lower and skip spaces
    # s = ''.join(o.lower() for o in s if o.isalnum())
    
    def is_alphanum(o):
        return (
            (ord('a') <= ord(o) <= ord('z')) or                
            (ord('A') <= ord(o) <= ord('Z')) or
            (ord('0') <= ord(o) <= ord('9'))
        )
    
    # setup pointers to the end
    l, r = 0, len(s) - 1
    
    # continue while the pointers don't cross
    while l < r:
        
        # skip over non-alphanum characters, as long as pointers don't cross
        # NOTE: this condition was tricky
        # Can't go until the end of the string, doesn't make sense
        # Only need to check if we overshoot the current pointer
        while (l < r) and not is_alphanum(s[l]):
            l += 1
        while (r > l) and not is_alphanum(s[r]):
            r -= 1
            
        # at this point, we are comparing two alphanumeric characters. Are they the same?
        if s[l].lower() != s[r].lower():
            return False
        
        # else, they were valid, move on to the next characters
        l += 1
        r -= 1
    
    # if we made it out here, the characters matches
    return True


    # there is an uglier version without the up-front cleaning up:
    # basically:
    '''
    Setup pointers
    While they don't cross:
        Skip over left non-alphanum in while loop (l < r); l+=1
        Skip over right non-alphanum in while loop (r > l) r-=1
        Check if current lower() is the same. If they're not, return False
            if s[l].lower() != s[r].lower():
                return False
        # advance the left pointer, decrement the right
        l, r = l + 1, r - 1
    
    Note: can check the alphanum status by ord(a|A|0) <= ord(x) <= ord(z|Z|9)
    '''
    

def longestPalindrome(self, s: str) -> str:
    '''Main idea:

    Ususally, palindromes are checked by working our way out from the string.
    But, we can also start from a character a move outward, as long as we're in bounds

    There is a small edge case for even vs. odd palindromes, which we can handle with two cases.

    At each point, while the pointers from a character are valid, we check incremental palindromes until we find the longer.

    The brute force solution is checking every substring for being a palindrome:
        n * n^^2 -> (check if pal) * (n^2 substrings for string of len n)
    '''
    
    # store the outputs
    res, res_len = '', 0

   
    # go through each character
    for i in range(len(s)):
        
        # check odd-len palindromes
        l, r = i, i
        while (0 <= l <= r < len(s)) and s[l] == s[r]:
            if (r - l + 1) > res_len:
                res = s[l:r+1]
                res_len = r - l + 1
            l -= 1
            r += 1
            
        # check even-len palindromes
        l, r = i, i + 1
        while (0 <= l < r < len(s)) and s[l] == s[r]:
            if (r - l + 1) > res_len:
                res = s[l:r+1]
                res_len = r - l + 1
            l -= 1
            r += 1
            
    return res

    '''Below slightly cleaned up to use a function'''
        # store the outputs
        res, res_len = '', 0

        # note: can avoid a decent amount of repeat work by passing in prev len
        def sweep_out_palindrome(beg, end, prev_len):
            cur_res, cur_len = '', prev_len
            l, r = beg, end
            while (0 <= l <= r < len(s)) and s[l] == s[r]:
                if (r - l + 1) > cur_len:
                    cur_res = s[l:r+1]
                    cur_len = r - l + 1
                l -= 1
                r += 1
            return cur_res

        # go through each character
        for i in range(len(s)):

            # check odd-len palindromes
            best_odd = sweep_out_palindrome(i, i, res_len)
            # check even-len palindromes
            best_even = sweep_out_palindrome(i, i + 1, res_len)

            # find the longest palindrome
            for pal in (best_odd, best_even):
                if len(pal) > len(res): res = pal
            res_len = len(res) # update the length

        return res


def countSubstrings(self, s: str) -> int:
    '''NOTE: don't get fancy.

    This is a modification of the problem above, finding the longest pal substring.
    Instead of keeping track of the max, simply return all valid palindrome substring.

    Check both even and odd.

    Then, return the combination of them.
    '''
    
    # store the outputs
    res = []

    # note: can avoid a decent amount of repeat work by passing in prev len
    def sweep_out_palindrome(beg, end):
        cur_res = []
        l, r = beg, end
        while (0 <= l <= r < len(s)) and s[l] == s[r]:
            cur_res.append(s[l:r+1])
            l -= 1
            r += 1
        return cur_res

    # go through each character
    for i in range(len(s)):

        # check odd-len palindromes
        odd_pals = sweep_out_palindrome(i, i)
        # check even-len palindromes
        even_pals = sweep_out_palindrome(i, i + 1)

        res.extend(odd_pals + even_pals)
        
    return len(res)

    '''Optimized version below, avoiding extra memory and using list comps + generator'''
        # store the outputs
        res = []

        # note: can avoid a decent amount of repeat work by passing in prev len
        def count_sub_palindromes(beg, end):
            cnt = 0
            l, r = beg, end
            while (0 <= l <= r < len(s)) and s[l] == s[r]:
                cnt += 1
                l -= 1
                r += 1
            return cnt

        # count up even and odd in one pass
        num_pals = [count_sub_palindromes(i, i) + count_sub_palindromes(i, i+1) for i in range(len(s))]
        # count them up and return
        return sum(num_pals)

        '''Lastly, avoid another function call and do the work in the inner function
        Functional and generator-based.
        '''
        # store the outputs
        res = []

        # note: can avoid a decent amount of repeat work by passing in prev len
        def count_sub_palindromes(pos):
            # stores local count of palindromes
            cnt = 0
            # count odd pals
            l, r = pos, pos
            while (0 <= l <= r < len(s)) and s[l] == s[r]:
                cnt += 1
                l -= 1
                r += 1
            # count even pals
            l, r = pos, pos + 1
            while (0 <= l <= r < len(s)) and s[l] == s[r]:
                cnt += 1
                l -= 1
                r += 1
            return cnt
        
        # count up even and odd palindromes in one go, then return total count
        return sum(count_sub_palindromes(i) for i in range(len(s)))


class Codec:
    '''Goal is to encode and decode a string.

    The problem with using a special character as delimeter, is we could have any combo of characters in the string. The longer our code, the less likely it is to appear in the sequence. But, we tradeoff memory and larger encoding space.

    The intuitive idea:
        Add some special info (PREFIX!!) to each string, that tells us how to decode it.
        Then, separate this info with our own delimeter
            We create the known context where it will occur, aka other random occurences in the string don't matter

        Then, when decoding the string, process the front into and incrementally read from the buffer

    '''
    _DELIM = "|"
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        enc = []
        for s in strs:
            enc.append(f'{len(s)}{self._DELIM}{s}')
        return ''.join(enc)
        # # cleaned up one-liner:
        # return ''.join(f'{len(s)}{self._DELIM}{s}' for s in strs)
        

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        # holds the decoded strings
        strs = []
        # start reading from the beginning
        i = 0

        # read until we finish the entire string
        while i < len(s):

            digits = [] # holds digits/counts of the current string

            # keep reading until we hit the delimeter
            j = i
            while s[j] != self._DELIM:
                digits.append(s[j])
                j += 1
            num_digits = int(''.join(digits))

            # read the string itself
            strs.append(s[j + 1: j + 1 + num_digits])

            # NOTE: at this point, we have to move i past the: len, delimeter, string itself
            # because of the non-inclusive range above, this points us to the next character
            i = j + 1 + num_digits
                
        return strs


'''Check if ransom note can be built from the characters in magazine

Basically checking if ransome note can be constructed from a subsequence of magazine'''
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        # build character map from magazine
        cmap = {}
        for c in magazine:
            cmap[c] = 1 + cmap.get(c, 0)
            
        # check if we can build the note
        for c in ransomNote:
            # if the character is not in the map at all, or its count is zero, return False
            if cmap.get(c, 0) == 0:
                return False
            # else, the character is at least in the map. consume it (decrement its count)
            cmap[c] -= 1
            # # can avoid this deletion step thanks to python's dict get() w/ default
            # if cmap[c] == 0:
            #     del cmap[c]
        
        # if we got here, we were able to build the string
        return True


'''Longest palindrome:

Key ideas:
    Use a set
    Think about nature of palindromes
    With even number of characters, would be able to build palindrome no problem
    With odd, we could insert a single character in the middle
        If |n| = 1, then it trivially grows the pal by 1
        if n = 3 or greater, we could splice off one to insert in the middle, AND
            Take remaining two and tack them on the side
    
    But, importantly for this last case, since we only track the instance of characters, no their count, when we subtract len(odd_chars), it's like we're "peeling off" this lone char.
        The 2n+ instances of the char are free to be peeled off from the string.

More detail:
A quick explanation:
Let's take a look at what our hash looks like in our test case.
"abccccdd"
{'a', 'b'}

You'll notice that it's only the characters that appear an odd amount of times in our string that get saved
This is because for all even occurrences of a letter we are guaranteed to be able to make a palindrome.

i.e.
ccddcc
dccccd
cdccdc

As you can see a length of 6 from the c's and d's can be acquired

So where does this plus 1 come into play?
len(s) - len(hash) + 1
In a purely even palindrome we can add back at most one odd occurrence of a letter

ex:
ccdadcc
ccdbdcc

if len(hash) == 0
This means all the characters occur an even amount of times in the string and the length of the string itself should be returned
ex.
"ccccdd" -> "ccddcc" length 6

Furthermore let's say that the that a and b were 3 characters long
"aaabbbccccdd"
in that case the hash would look exactly the same
since for every time we put an element in the set we remove it if we see it again only the odd occurring elements appear again.
{'a', 'b'}

as such our resulting palindrome would look something like this
"abccd" + "a" + "dccba" with either a or b as a possible middle character.
'''
class Solution:
    def longestPalindrome(self, s: str) -> int:
        # build character map of the strings
        cmap = {}
        for c in s: 
            cmap[c] = 1 + cmap.get(c, 0)
            
        # find the odd characters
        odd_chars = [c for c,v in cmap.items() if v & 1]
        
        # get even characters, aka we could take two to build the entire palindrome
        # plus, we could choose any of the odd character to insert in the middle
        npals = len(s) - len(odd_chars) + bool(odd_chars)
        
        return npals
        
