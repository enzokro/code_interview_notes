import heapq

class MedianFinder:
    def __init__(self):
        # Minimum heap to store the larger half of the stream
        self.min_heap = []
        # Maximum heap to store the smaller half of the stream
        self.max_heap = []

    def addNum(self, num):
        # If max heap is empty or num < current median, push -num onto max heap
        if not self.max_heap or -num > self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        # Otherwise, push num onto min heap
        else:
            heapq.heappush(self.min_heap, num)

        # If max heap has more elements, pop top element and push onto min heap
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        # If min heap has more elements, pop top element and push onto max heap
        elif len(self.min_heap) > len(self.max_heap) + 1:
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def findMedian(self):
        # If heaps have same size, return average of top elements
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        # Otherwise, return top element of heap with larger size
        elif len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        else: return self.min_heap[0]