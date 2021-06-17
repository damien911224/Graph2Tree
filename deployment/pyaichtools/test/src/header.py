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