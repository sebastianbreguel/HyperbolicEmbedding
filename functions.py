import random
import torch

def metrica(x, y):
	h = 0
	for i in range(min(len(x), len(y))):
		if x[i] == y[i]:
			h +=1
		else:
			break
	return len(x) + len(y) - 2 * h

print('metrica:', metrica('babba', 'bac'))

def generar_palabra(max_length=20, vocabuary=['a','b','c']):
	s = ''
	for _ in range(max_length):
		s += random.choice(vocabuary)
	return s

v = 3
l = 5

s = 'abcdefghijklmnopqrstuvwxyz'[:v]
print(s)