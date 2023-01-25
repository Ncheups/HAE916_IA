import sympy as sp

w0 = sp.symbols('w0')
w1 = sp.symbols('w1')
w2 = sp.symbols('w2')
w3 = sp.symbols('w3')
w4 = sp.symbols('w4')
w5 = sp.symbols('w5')
w6 = sp.symbols('w6')
w7 = sp.symbols('w7')

x1 = sp.symbols('x1')
x2 = sp.symbols('x2')

h1in = w0*x1 + w2*x2
h2in = w1*x1 + w3*x2

t1 = sp.symbols('t1')
t2 = sp.symbols('t2')

h1out = 1/(1+sp.exp(-h1in))
h2out = 1/(1+sp.exp(-h2in))

y1in = w4*h1out + w6*h2out
y2in = w5*h1out + w7*h2out

y1out = 1/(1+sp.exp(-y1in))
y2out = 1/(1+sp.exp(-y2in))

L = pow(y1out - t1,2) + pow(y2out - t2, 2)

gradL0 = sp.diff(L, w0)
gradL1 = sp.diff(L, w1)
gradL2 = sp.diff(L, w2)
gradL3 = sp.diff(L, w3)
gradL4 = sp.diff(L, w4)
gradL5 = sp.diff(L, w5)
gradL6 = sp.diff(L, w6)
gradL7 = sp.diff(L, w7)