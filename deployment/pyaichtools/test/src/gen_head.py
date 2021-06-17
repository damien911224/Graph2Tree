try:
	result = 0
except Exception:
	print("Timed out", sys.exc_info()[0])
	result = 0
if type(result) == float:
	if int(result) == result:
		result = int(result)
	else:
		result = "{:.2f}".format(round(result, 2))