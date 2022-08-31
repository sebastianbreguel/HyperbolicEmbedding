import random
import torch
import pandas as pd
import numpy as np

v = 6
l = 10
n = 1000000

s = 'abcdefghijklmnopqrstuvwxyz'[:v]
vocabuary = [i for i in s]

def metrica(x, y):
	h = 0
	for i in range(min(len(x), len(y))):
		if x[i] == y[i]:
			h +=1
		else:
			break
	return len(x) + len(y) - 2 * h

def generar_palabra(max_length=20, vocabuary=['a','b','c']):
	s = ''
	for _ in range(random.randint(0, max_length)):
		s += random.choice(vocabuary)
	return s

lista = []

for _ in range(n):
	a = generar_palabra(l, vocabuary)
	b = generar_palabra(l, vocabuary)
	lista.append([a, b, metrica(a, b)])

df = pd.DataFrame(lista)

print(df)