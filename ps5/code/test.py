def myfun(b, c):
    print(b, c)

def hello2(a, kwargs):
    myfun(**kwargs)
    pass

def hello(a, kwargs):
    # myfun(kwargs)
    hello2(a, kwargs=kwargs)
    pass

# lol = {'b': 2, 'c': 3}
hello(myfun, kwargs={'b': 2, 'c': 3})
