for var1, var2 in itertools.combinations([QL[0], QL[1], QL[2], QL[3], QL[4]], const2):
    if var1 > QL[2]:
        var0+=(math.acos(const1) + math.atan(var2))
result = var0