class TimedLL:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, src, dst, t):
        new_node = TimedLLNode(src, dst, t)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
        
    def popleft(self):
        if not self.head:
            return None
        node = self.head
        self.head = self.head.next
        if not self.head:
            self.tail = None
        else:
            self.head.prev = None
        self.size -= 1
        return node.src, node.dst, node.t
    
    def remove_node(self, node):
        if self.head == node:
            self.head = node.next
        if self.tail == node:
            self.tail = node.prev
        node.remove()
        self.size -= 1

class TimedLLNode:
    def __init__(self, src, dst, t):
        self.src = src
        self.dst = dst
        self.t = t
        self.next = None
        self.prev = None
    
    def remove(self):
        if self.prev:
            self.prev.next = self.next
        if self.next:
            self.next.prev = self.prev
        self.next = None
        self.prev = None
    
