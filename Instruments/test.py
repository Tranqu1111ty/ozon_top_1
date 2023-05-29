

def gg(bb: int):
    global f1
    f1 = bb

    return 0

def tt(qq: int):

    gg(qq)

    return f1



for i in range(50):
    print(tt(i))