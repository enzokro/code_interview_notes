

def insertNewInterval(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    '''Lots of interval math.

    The "base cases" are easy but need to think through the logic more'''
    
    # stores the output
    res = []

    # destructure the new interval
    new_beg, new_end = newInterval
    
    # iterate through the intervals
    for idx, (beg, end) in enumerate(intervals):

        # Case 1): New interval is fully before this current one
        # NOTE: at this point, newInterval could have been updated
        if newInterval[-1] < beg:
            res.append((new_beg, new_end))
            # The same will be true for all other intervals, so can return at this point
            # NOTE: thanks to sorting. Possible to have grown and merged other earlier intervals at this point
            return res + intervals[idx:]
            
        # Case 2): New interval is fully after the current one
        elif newInterval[0] > end:
            res.append((beg,end))
            # NOTE: because intervals are sorted, there are others we could match with
            # have to continue our search
            
        # Case 3): Interval overlaps partially into current
        else:
            new_beg, new_end = min(new_beg, beg), max(new_end, end)
            # NOTE: found a new potential interval, but we can't add it cause there could be other overlaps
            # have to continue marching and iterating
            # if there are more overlaps, the interval will continue growing
            # else, will continue until one of the other checks completes
        
    # sanity check that we actually insert the new interval
    # NOTE: we need this because it is possible for the interval to keep growing, but for it to never be fully-before the current one
    # Then, case 1) never executes and we have a hanging new-interval that's never added
    # If case 1) ever executes, we return directly so we are good to go
    # As for case 2), it is possible to add all other intervals that come before our new meged interval
    # But, because we need to check for all possible post-orders, we don't add the new interval
    res.append([new_beg, new_end])
            
    return res

    '''Much cleaner version showing intuitive idea of: adding earlier intervals, merging the overlaps, adding intervals afterward'''
    n = len(intervals)
    i, output = 0, []
    
    # Append first non-overlapping intervals
    while i < n and intervals[i][1] < newInterval[0]:
        output.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals (if no overlapping then just append newInterval)
    while i < n and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(intervals[i][0], newInterval[0])
        newInterval[1] = max(intervals[i][1], newInterval[1])
        i += 1
    output.append(newInterval)
    
    # Append remaining non-overlapping intervals
    while i < n:
        output.append(intervals[i])
        i += 1
        
    return output

    '''Same idea as above with more commented code about the cases'''
    result = []
    
    for interval in intervals:
        # the new interval is after the range of other interval, so we can leave the current interval because the new one does not overlap with it
        if interval[1] < newInterval[0]:
            result.append(interval)
        # the new interval's range is before the other, so we can add the new interval and update it to the current one
        elif interval[0] > newInterval[1]:
            result.append(newInterval)
            newInterval = interval
        # the new interval is in the range of the other interval, we have an overlap, so we must choose the min for start and max for end of interval 
        elif interval[1] >= newInterval[0] or interval[0] <= newInterval[1]: # NOTE: can technically be `else`
            newInterval[0] = min(interval[0], newInterval[0])
            newInterval[1] = max(newInterval[1], interval[1])

    # same trick of making sure the new interval is added in 
    result.append(newInterval); 
    return result


def mergeIntervals(self, intervals: List[List[int]]) -> List[List[int]]:
    '''Entry point into interval problems. 


    Can get creative and think about insertion points and what to do.
    '''
    # edge cases to avoid sorting
    if len(intervals) <= 1:
        return None or intervals
    
    # sort the interval by starting pos
    intervals.sort(key=lambda o: o[0])
    
    # NOTE: this was likely getting me
    # start with the first interval to avoid an edge case (which one?)
    res = [intervals[0]]
    
    # step through the rest of the intervals
    for beg, end in intervals[1:]:
        
        #grab the end of the most recently added interval
        prev_end = res[-1][-1]
        
        # check if it overlaps with the current interval, aka we have to merge
        if beg <= prev_end:
            # can merge by changing the end value to the max of the two
            # [1, 5], [2, 4] -> merged into [1, max(5,4)] -> [1, 5]
            # the start point stays the same, we don't have to change it. 
            # ^ BUT! Only because we sorted them by starting point, so can guaranteee that the beg of our prev is before the beg of the current
            res[-1][-1] = max(prev_end, end)
        else:
            res.append([beg, end])
            
    return res


    '''Concise version of above, avoiding the base case checks at the start.
    And, avoiding taking the first interval with some clever logic in the first check.
    '''
    # check for empty intervals
    if not intervals: return None
    # sort them by starting point
    intervals.sort(key=lambda x: x[0])

    # start with the first interval
    merged = [intervals[0]]
    # iterate through the remaining ones
    for beg, end in intervals[1:]:
        # if the current interval does not overlap with the previous, simply append it.
        if beg > merged[-1][-1]:
            merged.append([beg,end])
        else:
        # otherwise, there is overlap, so we merge the current and previous intervals.
            merged[-1][-1] = max(merged[-1][-1], end)

    return merged

    '''Super clean version, but needed a few edits since inputs are now lists instead of classes'''
    out = []
    for i in sorted(intervals, key=lambda i: i[0]):
        if out and i[0] <= out[-1][-1]:
            out[-1][-1] = max(out[-1][-1], i[-1])
        else:
            out.append(i) # += i, <- allegedly faster? need to check compiled code
    return out
        

def removeIntervals(intervals):
    '''Drop the minimum number of intervals to make all intervals non-overlapping.'''
    # sort them by starting point
    intervals.sort() # first by start point, then end point in case of a tie

    # grab the first end point
    prev_end = intervals[0][-1]
    # number of intervals to remove
    to_remove = 0

    # walk through the remaining intervals
    for (beg, end) in intervals[1:]:

        # check if these overlap
        if beg < prev_end:
            # if they do, we need to remove one of them
            to_remove += 1

            # how to remove? update the previous end
            # keep the one with the minimum end value
            # cuts the intervals as short as possible to avoid the overlap
            prev_end = min(prev_end, end)

        # they don't overlap, move up our most-recent end
        else: # beg >= prev_end
            prev_end = end

    return to_remove


def meetingRooms(intervals):
    '''Given a set of meeting start and stop times, can a person attend them?

    Basically checking if there are any intervals that overlap.
    As usual, it helps to sort them by starting time.
    '''
    if len(intervals) <= 1: return True # can technically attend no meetings, or one
    
    # sort the intervals by starting time
    intervals.sort(key= lambda o: o[0])
    
    # start with the first meeting
    prev_end = intervals[0][-1]
    
    # check other meetings
    for (beg, end) in intervals[1:]:
        
        # do these overlap?
        # aka: does this meeting begin before the previous one ended?
        if prev_end > beg:
            return False
        
        # if they don't overlap, we could attend this meeting and prepare to check to the next one
        else:
            prev_end = end
            
    return True


def meetingRooms2(intervals):
    '''Similar to above, but now we need to find how many rooms to hold all meetings.

    Basically, what is the largest amount of concurrent meetings throuhgout the schedule?

    We need two pointers: one for beginning times and the other for end times
    '''
    # edge case: 0 or 1 meetings
    if len(intervals) <= 1: return len(intervals)

    # two lists, one for start time and the other for end times
    begs = [o[0] for o in intervals]
    ends = [o[1] for o in intervals]
    # sort them both
    begs.sort(), ends.sort()

    # setup two pointers: one for beginning and the other for ends
    bi, ei = 0, 0

    # local and global meeting room counts
    cur_rooms, total_rooms = 0, 0

    # NOTE: the starting point will finish before the end array
    while bi < len(intervals):

        # does the current meeting start while another one hasn't ended?
        if begs[bi] < ends[ei]:
            # if yes, we need more rooms, and we can check the start time of the next meeting
            cur_rooms += 1
            bi += 1

        # else, the meeting has ended, and a room frees up
        else:
            cur_room -= 1
            ei += 1

        # now, check whether the local amount of meetings has grown
        total_rooms = max(cur_rooms, total_rooms)

    # return the maximum number of concurrent meetings
    return total_rooms


