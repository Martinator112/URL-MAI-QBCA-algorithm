import heapq

class MaxHeapObj(object):
  def __init__(self, val, priority): 
    self.val = val
    self.priority = priority
  def __lt__(self,other): return self.priority > other.priority
  def __eq__(self,other): return self.priority == other.priority
  def __str__(self): return str(self.val)

class MinHeap(object):
  def __init__(self): self.h = []
  def heappush(self,x): heapq.heappush(self.h,x)
  def heappop(self): return heapq.heappop(self.h)
  def __getitem__(self,i): return self.h[i]
  def __len__(self): return len(self.h)
  def is_empty(self): return len(self) == 0

class MaxHeap(MinHeap):
  def heappush(self,x,p): heapq.heappush(self.h,MaxHeapObj(x,p))
  def heappop(self): return heapq.heappop(self.h)
  def __getitem__(self,i): return self.h[i]
