import math
import itertools
import signal
import sys
var0 = 0
var1 = 0
var2 = 0
var3 = 0
var4 = 0
const0 = 0
const1 = 1
const2 = 2
result = None
def signal_handler(signum, frame):
    raise Exception("Timed out!")
signal.signal(signal.SIGALRM, signal_handler)
signal.alarm(1)
QL = [0.1, 0.2, 0.3, 0.4, 0.5]
NL = ["11"]
try:
    result = 0
    result = QL[1] * QL[1]
except Exception:
    print("Timed out", sys.exc_info()[0])
    result = 0
if type(result) == float:
    if int(result) == result:
        result = int(result)
    else:
        result = "{:.2f}".format(round(result, 2))
print('recovered source output:', result)
