def transpose_mat(mat):
    return list(zip(*mat))
def rotate_mat_90(mat):
    return list(zip(*mat[::-1]))
def rotate_mat_270(mat):
    return list(zip(*mat))[::-1]
def rotate_mat_180(mat):
    return rotate_mat_90(rotate_mat_90(mat))



class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        '''Nice little graph traversal algorithm. Given a starting point, paint all adjacent cells with same starting value the new, given `color`
        
        Classic: keep visited set, get matrix dimensions.
        BFS with the condition that the current cell has the same, starting color as the original
            We don't want to miss a cell because we've already change it to `color`
        '''
        
        # setup variables:
        visited = set() # to avoid re-visiting cells
        deltas = [0, 0, -1, 1] # to visit adjacent cells
        nrows, ncols = len(image), len(image[0]) # image dimensions
        
        def paint_matrix(r, c, prev_color):
            
            # check current pixel if we haven't visited it yet
            if (0 <= r < nrows) and (0 <= c < ncols) and (r,c) not in visited:
                # mark node as visited
                visited.add((r,c))
                
                # check if we need to paint this pixel
                if image[r][c] == prev_color:
                    image[r][c] = color
                    
                    # paint its neighbors
                    for dr, dc in zip(deltas, deltas[::-1]):
                        paint_matrix(r + dr, c + dc, prev_color)
                    
        # paint and return the matrix
        paint_matrix(sr, sc, image[sr][sc])
        return image
                

class Solution:
    # idea:
    #   each row in input array is independent and can be processed separately
    # how to process each row:
    #   we need to move stones ("#") to empty spaces (".") from left to right
    #   since it's only move from left to rigth, we can iterate from the end of the row
    #   and keep in memory the last non-obstacle space where we can move stones
    # and at the end we just need to rotate array
    
    def rotateTheBox(self, box: List[List[str]]) -> List[List[str]]:
        for row in box:
            move_position = len(row) - 1             # initialize with the last position in row
            for j in range(len(row) - 1, -1, -1):    # iterate from the end of the row
                if row[j] == "*":                    # we cannot move stones behind obstacles,
                    move_position = j - 1            # so update move position to the first before obstacle
                elif row[j] == "#":                  # if stone, move it to the "move_position"
                    row[move_position], row[j] = row[j], row[move_position]
                    move_position -= 1

        return zip(*box[::-1])                       # rotate array


class Solution:
    def getBiggestThree(self, grid: List[List[int]]) -> List[int]:
        
        l=[] # stores the sums
        n=len(grid) # number of rows
        m=len(grid[0]) # number of cols
        
        for i in range(n):
            for j in range(m):
                # iterate over every tile
                
                # top tile point of rhombus
                ans=grid[i][j]
                l.append(grid[i][j]) # valid rhombus sum (one point)
                
                # distance var to store distance from j to both ends of rhombus
                # (used to increase size of rhombus)
                distance=1 
                
                # make sure the tile is within grid bounds
                while(i+distance<n and j-distance>=0 and j+distance<m):
                    # iterate over all possible rhombus sizes using the distance var
                    
                    a=i+distance # next row
                    b=j+distance # col to the right
                    c=j-distance # col to the left
                    
                    # right tile point of rhombus:  grid[a][b]
                    # left tile point of rhombus: grid[a][c]
                    ans+=grid[a][b]+grid[a][c]
                    
                    # a dummy variable to store the present sum of the sides
                    # (left and right tile point)
                    dumm=0    
                    while(True):
                        # iterate to find the bottom point of rhombus
                        
                        c+=1 # left tile point moves toward the right
                        b-=1 # right tile point moves toward the left
                        a+=1 # moves to the bottom (next row)
                        
                        if(c==m or b==0 or a==n):
                            break # reached bounds
                        
                        # left and right cols met at "middle"
                        if(c==b): # found the bottom tile point of rhombus
                            # add bottom tile sum to sides (left and right) sum
                            dumm+=grid[a][b] 
                            l.append(ans+dumm) #appending the obtained sum
                            break
                            
                        dumm+=grid[a][b]+grid[a][c] # adding both sides sum to dummy
                        
                    distance+=1
                    
        l=list(set(l)) # remove duplicates
        l.sort(reverse=True) 
        # return first 3 largest sums
        return l[:3]



def rotateMatrix(matrix):
    '''Need to rotate the matrix by 90 degrees.

    With math, could do with cos/sin matrix.

    It has the same pattern as spiral matrix:
        Four pointers: top/bottom, left/right
        With a bit of ease since we have a square matrix

    Swap elements, saving the one being overwritten, one row/column at a time.
        Need to swap n - 1 element since the start/end elements are swapped

    Then we work our way in until the pointers overlap, just like in spiral matrix
    '''
    # left and right pointers
    l, r = 0, len(matrix) - 1
    
    # continue until the pointers don't overlap
    while l < r:
        # we have to move (n-1) elements
        for i in range(r - l):
            # set top/bottom pointers
            top, bottom = l, r
            
            # save the top-left value we're above to overwrite
            top_left = matrix[top][l + i] # add i to offset the elements needed
            
            # move bottom left into top-left
            matrix[top][l + i] = matrix[bottom - i][l]
            
            # move bottom right into bottom-left
            matrix[bottom - i][l] = matrix[bottom][r - i]
            
            # move top-right into bottom-right
            matrix[bottom][r - i] = matrix[top + i][r]
            
            # finally, top left into top right
            matrix[top + i][r] = top_left # move the value we solved
            
            # # one-liner without temp variable
            # matrix[top][l + i], matrix[bottom - i][l], matrix[bottom][r - i], matrix[top + i][r] = \
            #     matrix[bottom - i][l], matrix[bottom][r - i], matrix[top + i][r], matrix[top][l + i]
            
        # one last thing, update pointers
        r -= 1
        l += 1


class Solution:
    def rotate(self, A):
        n = len(A)
        for i in range(n/2):
            for j in range(n-n/2):
                A[i][j], A[~j][i], A[~i][~j], A[j][~i] = \
                         A[~j][i], A[~i][~j], A[j][~i], A[i][j]

class Solution:
    def rotate(self, A):
        A[:] = list(map(list, zip(*A[::-1])))

class Solution:
    def rotate(self, A):
        A[:] = [[row[i] for row in A[::-1]] for i in range(len(A))]


class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        '''Based on insight that values on same diagonal have index pairs with the same sum
        '''
        d = collections.defaultdict(list)
        
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                d[i+j].append(mat[i][j])
                
        out = []
        
        for k,v in d.items():
            out.extend(v if k & 1 else v[::-1])
            
        return out


class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        if r * c != m * n: return mat  # Invalid size -> return original matrix
        ans = [[0] * c for _ in range(r)]
        for i in range(m * n):
            ans[i // c][i % c] = mat[i // n][i % n]
        return ans

class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        res = [[None for _ in range(c)] for _ in range(r)]
        if r*c!=len(mat)*len(mat[0]):
            return mat
        count=0
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                res[count//c][count%c]= mat[i][j]
                count+=1
        return res

class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        if r*c!=len(mat)*len(mat[0]):
            return mat
        queue = [cell for row in mat for cell in row]
        return [[queue.pop(0) for _ in range(c)] for _ in range(r)]


class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        for i in range(len(matrix) - 1):
            for j in range(len(matrix[0]) - 1):
                if matrix[i][j] != matrix[i + 1][j + 1]:
                    return False
        return True

        

def setZeroes(self, matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.

    Worst approach does > m * n work by copying the input and setting entire row/col whenever a zero cell is reached.

    Slightly better to keep a set/list of row/cols that need to be zero, memory goes to m + n

    But, we can re-use the top row itself as our buffer, with one aditional memory cell for the points where they overlap.
    """

    '''First approach uses extra memory to keep track of cells/rows that should be zerod out.
    We incur more memory from saving the rows/cols that will be zerod out
    '''
    nrows, ncols = len(matrix), len(matrix[0])
    if not (nrows or ncols): return
    deltas = [0,0,-1,1]
    
    rows, cols = set(), set()
    
    def has_zeros(r, c):
        if matrix[r][c] == 0:
            rows.add(r), cols.add(c)
    # find cells and rows we will need to zero out                    
    for r in range(nrows):
        for c in range(ncols):
            has_zeros(r,c)

    # iterate through the sets and set them clear
    for r in range(nrows):
        for c in range(ncols):
            if r in rows or c in cols:
                matrix[r][c] = 0


    '''Better approach with constant memory'''
    rowZero = False # could also be col
    NR, NC = len(matrix), len(matrix[0])

    # find which cells need to be zero
    for r in range(NR):
        for c in range(NC):
            if matrix[r][c] == 0:
                # mark the row and potentially column for annhihilation
                matrix[0][c] = 0 # <- set the first row in the column as zero
                # avoid the overlap cell, in case the first row needs zeroing
                if r == 0:
                    rowZero = True
                else:
                    matrix[r][0] = True # <- mark the first row as needing to be zerod

    # set the marked rows/cells to zero:
    for r in range(1, NR):
        for c in range(1, NC):
            # if they're not both 1, means we must zero this value
            if matrix[0][c] == 0 or matrix[r][0] == 0:
                matrix[r][c] = 0 

    # handle the first column needing to be zero (form top-left cell)
    if matrix[0][0] == 0:
        for r in range(NR):
            matrix[r][0] = 0

    # likewise for the row, if the flag was set
    if rowZero:
        for c in range(NC):
            matrix[0][c] = 0

                
def spiralOrder(matrix):
        
        '''Main idea:
        
        Have four pointer: top / bottom and right left as we step through the matrix.
        Then, while the pointers are in bounds:
            Clear the top row
            Clear the right column
            Clear the bottom row
            Clear the left column
            
        Main tricky points are index notation and edge cases for when the pointers cross over
        
        Write out and think about the pointers, note the in-place when we're going through a row or col, respectively.
        '''
        

        # setup response and grab pointers
        res = []

        left, right = 0, len(matrix[0])
        top, bottom = 0, len(matrix)

        # continue while pointers are valid
        while left < right and top < bottom:

            # add top row
            for i in range(left, right): # columns so far
                res.append(matrix[top][i])
            # res.extend(matrix[top][i] for i in range(left, right))
            top += 1

            # add right col
            for i in range(top, bottom):
                res.append(matrix[i][right - 1])
            # res.extend(matrix[i][right-1] for i in range(top, bottom))
            right -= 1

            # check for overshoot
            if left >= right or top >= bottom:
                break

            # add bottom row
            for i in range(right - 1, left - 1, -1):
                res.append(matrix[bottom-1][i])
            # res.extend(matrix[bottom - 1][i] for i in range(right - 1, left -1, -1))
            bottom -= 1

            # add left col
            for i in range(bottom - 1, top -1, -1):
                res.append(matrix[i][left])
            # res.extend(matrix[i][left] for i in range(bottom -1, top -1, -1))
            left += 1

        return res


class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:

        matrix = [[0] * n for _ in range(n)]

        left, right = 0, len(matrix[0])
        top, bottom = 0, len(matrix)

        val = 1

        while left < right and top < bottom:
            # populate the left row
            for i in range(left, right):
                matrix[top][i] = val
                val += 1
            # MOVE DOWN THE TOP ROW
            top += 1

            # pop the right col
            for i in range(top, bottom):
                matrix[i][right - 1] = val
                val += 1
            # MOVE DOWN THE LAST COL
            right -= 1

            # bounds check
            if left >= right or top >= bottom:
                break

            # pop the bottom row
            for i in range(right-1, left-1, -1):
                matrix[bottom-1][i] = val
                val += 1
            # MOVE UP THE BOTTOM ROW
            bottom -= 1

            # pop the left col
            for i in range(bottom -1, top-1, -1):
                matrix[i][left] = val
                val += 1
            # MOVE IN THE LEFT COL
            left += 1

        return matrix

