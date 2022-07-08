class Avger:
    def __init__(self, init=0):
        self.init = init
        self.value = init
        self.cnt = 0
    
    def __str__(self):
        return str(self.get())
    
    def reset(self):
        self.value = self.init
        self.cnt = 0
    
    def step(self, val):
        self.value += val
        self. cnt += 1
    
    def get(self):
        return (self.value / self.cnt) if self.cnt > 0 else self.init