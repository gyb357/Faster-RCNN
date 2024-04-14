def operate(a: bool, b, c):
    return b if a is True else c

def operate_elif(a: bool, b, c: bool, d, e):
    return b if a is True else operate(c, d, e)

