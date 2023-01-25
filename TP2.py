import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 

data = pd.read_csv('NNTraining.data')
data4 = pd.read_csv('NNTest.data')

def sigmoid(z):
    return 1/(1+math.exp(-z))

def Diff_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def Forward_Back(x1, x2, t1, t2, w):
    #Forward pass
    
    w0 = w[0]
    w1 = w[1]
    w2 = w[2]
    w3 = w[3]
    w4 = w[4]
    w5 = w[5]
    w6 = w[6]
    w7 = w[7]
    
    h1in = w[0]*x1 + w[2]*x2
    h2in = w[1]*x1 + w[3]*x2

    h1out = sigmoid(h1in)
    h2out = sigmoid(h2in)

    y1in = w[4]*h1out + w[6]*h2out
    y2in = w[5]*h1out + w[7]*h2out

    y1out = sigmoid(y1in)
    y2out = sigmoid(y2in)

    L = pow(y1out-t1,2) + pow(y2out-t2,2)

    #Backpropagation
    gradW = np.array([[0., 0., 0., 0., 0., 0., 0., 0.]])

    #gradW[0,4] = h1out*dg1*g1
    gradW[0,0] = 2*w4*x1*(-t1 + 1/(np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1)) + 1))*np.exp(-w0*x1 - w2*x2)*np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1))/((np.exp(-w0*x1 - w2*x2) + 1)**2*(np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1)) + 1)**2) + 2*w5*x1*(-t2 + 1/(np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1)) + 1))*np.exp(-w0*x1 - w2*x2)*np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1))/((np.exp(-w0*x1 - w2*x2) + 1)**2*(np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1)) + 1)**2)
    gradW[0,1] = 2*w6*x1*(-t1 + 1/(np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1)) + 1))*np.exp(-w1*x1 - w3*x2)*np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1))/((np.exp(-w1*x1 - w3*x2) + 1)**2*(np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1)) + 1)**2) + 2*w7*x1*(-t2 + 1/(np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1)) + 1))*np.exp(-w1*x1 - w3*x2)*np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1))/((np.exp(-w1*x1 - w3*x2) + 1)**2*(np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1)) + 1)**2)
    gradW[0,2] = 2*w4*x2*(-t1 + 1/(np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1)) + 1))*np.exp(-w0*x1 - w2*x2)*np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1))/((np.exp(-w0*x1 - w2*x2) + 1)**2*(np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1)) + 1)**2) + 2*w5*x2*(-t2 + 1/(np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1)) + 1))*np.exp(-w0*x1 - w2*x2)*np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1))/((np.exp(-w0*x1 - w2*x2) + 1)**2*(np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1)) + 1)**2)
    gradW[0,3] = 2*w6*x2*(-t1 + 1/(np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1)) + 1))*np.exp(-w1*x1 - w3*x2)*np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1))/((np.exp(-w1*x1 - w3*x2) + 1)**2*(np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1)) + 1)**2) + 2*w7*x2*(-t2 + 1/(np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1)) + 1))*np.exp(-w1*x1 - w3*x2)*np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1))/((np.exp(-w1*x1 - w3*x2) + 1)**2*(np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1)) + 1)**2)
    gradW[0,4] = 2*(-t1 + 1/(np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1)) + 1))*np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1))/((np.exp(-w0*x1 - w2*x2) + 1)*(np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1)) + 1)**2)
    gradW[0,5] = 2*(-t2 + 1/(np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1)) + 1))*np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1))/((np.exp(-w0*x1 - w2*x2) + 1)*(np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1)) + 1)**2)
    gradW[0,6] = 2*(-t1 + 1/(np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1)) + 1))*np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1))/((np.exp(-w1*x1 - w3*x2) + 1)*(np.exp(-w4/(np.exp(-w0*x1 - w2*x2) + 1) - w6/(np.exp(-w1*x1 - w3*x2) + 1)) + 1)**2)
    gradW[0,7] = 2*(-t2 + 1/(np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1)) + 1))*np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1))/((np.exp(-w1*x1 - w3*x2) + 1)*(np.exp(-w5/(np.exp(-w0*x1 - w2*x2) + 1) - w7/(np.exp(-w1*x1 - w3*x2) + 1)) + 1)**2)
    #Update
    for i in range(8):
        w[i] = w[i] - 0.01*gradW[0,i]
    
    t = np.array([[y1out, y2out]])

    return L, gradW.T, t


if __name__ == '__main__':
    
    x1 = np.array([[2],[1]])
    t1 = np.array([[1],[0]])
    
    x2 = np.array([[-1],[3]])
    t2 = np.array([[0],[1]])
    
    x3 = np.array([[1],[4]])
    t3 = np.array([[1],[0]])
    
    w = np.array([[2],[-3],[-3],[4],[1],[-1],[0.25],[2]])
    
    L1,gradW1, out1= Forward_Back(x1[0,0], x1[1,0],t1[0,0], t1[1,0],w)
    L2,gradW2, out2= Forward_Back(x2[0,0], x2[1,0],t2[0,0], t2[1,0],w)
    #L3,gradW3 = Forward_Back(x3,t3,w)
    
    w_for = w
    p = 2
    Niter = 100
    
    plt.figure(3)
    plt.title("1 Error p=2")
    plt.xlabel("Iteration Number")
    plt.ylabel("Error")
    
    for i in range(0, Niter):
        L1_for, grad_for_1, out1_for = Forward_Back(x1[0,0], x1[1,0], t1[0,0], t1[1,0], w_for)
        L2_for, grad_for_2, out2_for = Forward_Back(x2[0,0], x2[1,0], t2[0,0], t2[1,0], w_for)
        
        grad_for_mean = (grad_for_1 + grad_for_2)/2
        
        w_for -= p*grad_for_mean
        
        L_for_mean = (L1_for + L2_for)/2
        
        plt.plot(i,L_for_mean, 'b .')
    
    L3,gradW3, out3 = Forward_Back(x3[0,0], x3[1,0], t3[0,0], t3[1,0],w_for)
    
    
    
    ''' Partie 3 - 200pts ''' 
    
    x1 = data.iloc[:,0]
    x2 = data.iloc[:,1]
    t1 = data.iloc[:,2]
    t2 = data.iloc[:,3]
    
    p=2
    plt.figure()
    plt.title("Error p=2 refresh 6")
    plt.xlabel("Iteration Number")
    plt.ylabel("Error")
    plt.grid()
    
    w_for = w
    for i in range(0,10):
        grad_for_1_mean = 0
        grad_for_mean   = 0
        L_for_mean_1    = 0
        L_for_mean      = 0
        for j in range(0, np.size(x1, axis=0)):
            L1_for, grad_for_1, out1_for = Forward_Back(x1[j], x2[j], t1[j], t2[j], w_for)
            grad_for_1_mean += grad_for_1
            L_for_mean_1 += L1_for
        
            if (j+1)%6 == 0 and j > 0 :
                grad_for_mean = grad_for_1_mean/ (j+1)
                w_for -= p*grad_for_mean
                w_save = w_for
                L_for_mean = L_for_mean_1 / (j+1)
            
        plt.plot(i,L_for_mean, 'b .')
                
    
    #L3,gradW3, out3 = Forward_Back(x3[0,0], x3[1,0], t3[0,0], t3[1,0],w_for)
    
    ''' Partie 4 - 200pts ''' 
    
    x1 = data4.iloc[:,0]
    x2 = data4.iloc[:,1]
    t1 = data4.iloc[:,2]
    t2 = data4.iloc[:,3]
    
    
    w = np.array([[2],[-3],[-3],[4],[1],[-1],[0.25],[2]])
    L4_base = np.zeros([np.size(x1, axis=0), 1])
    for sample in range (0, np.size(x1, axis=0)):
        L4_base[sample, 0], grad_base, out4_base = Forward_Back(x1[sample], x2[sample], t1[sample], t2[sample], w)
    print(L4_base[99, 0])
    print(w)
    
    plt.figure(2)
    plt.title("Before training")
    plt.xlabel("Iteration Number")
    plt.ylabel("Error")
    plt.grid()
    plt.plot(L4_base)
    
    L4 = np.zeros([np.size(x1, axis=0), 1])
    for sample in range (0, np.size(x1, axis=0)):
        L4[sample, 0], grad, out4 = Forward_Back(x1[sample], x2[sample], t1[sample], t2[sample], w_save)
    print(L4[99, 0])
    print(w_for)
    
    plt.figure(1)
    plt.title("After training")
    plt.xlabel("Iteration Number")
    plt.ylabel("Error")
    plt.grid()
    plt.plot(L4)
    

    
    plt.show()