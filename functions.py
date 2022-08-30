def metrica(x, y):
	h = 0
	for i in range(min(len(x), len(y))):
		if x[i] == y[i]:
			h +=1
		else:
			break
	return len(x) + len(y) - 2 * h
print('metrica:', metrica('babba', 'bac'))

v = 3
l = 5

s = 'abcdefghijklmnopqrstuvwxyz'[:v]
print(s)