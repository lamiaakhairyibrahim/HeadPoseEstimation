class Queue:
    def __init__(self , max_size):
        self.max_size = max_size
        self.queue = []
        self.length = 0

    def enqueue(self,x):
        if self.length < self.max_size :
            self.queue.append(x)
            self.length = self.length+1
        else:
            print("You have reached the maximum size") 
    def dequeue(self ):
        if len(self.queue) > 0 :
            remove = self.queue[0]
            del self.queue[0]
            self.length = self.length-1
            return remove
        else:
            print("Queue is Empty")
    def isfull(self):
        if self.length == self.max_size:
            return True
        
    def print_queu(self):
        for i in self.queue:
            print(i)