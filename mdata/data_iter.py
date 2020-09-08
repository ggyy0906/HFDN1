def inf_iter(iterable, with_end=False):
    while True:
        for i in iterable:
            yield i
        if with_end:
            yield None

class EndlessIter:

    """ A helper class which iter a dataloader endnessly
    
    Arguments:
        dataloader {DataLoader} -- Dataloader want to iter
    
    """

    def __init__(self, dataloader, max = None):
        assert dataloader is not None
        self.l = dataloader
        self.it = iter(dataloader)
        self.current_iter = 0
        self.max_iter = max

    def next(self):
        """ return next item of dataloader, if use 'endness' mode, 
        the iteration will not stop after one epoch
        
        Keyword Arguments:
            need_end {bool} -- weather need to stop after one epoch
             (default: {False})
        
        Returns:
            list -- data
        """

        # return None when reach max iter
        if self.current_iter == self.max_iter:
            self.current_iter = 0
            return None
        
        try:
            i = next(self.it)
        except Exception:
            self.it = iter(self.l)
            # continue to next epoch when max iter is None
            if self.max_iter is None:
                i = next(self.it)
            # if max iter sets to -1, return None when ends an epoch 
            elif self.max_iter <= 0:
                i = None
            else:
                i = next(self.it)

        self.current_iter += 1
        return i        


if __name__ == "__main__":
    targets = list(range(15))
    it = EndlessIter(targets, max=10)

    while True:
        c = it.next()
        if c is None:
            print('ends')
            break
        else:
            print(c)
    
    while True:
        c = it.next()
        if c is None:
            print('ends')
            break
        else:
            print(c)

