# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 01:47:40 2016

@author: puneeth
"""

import numpy as np
import re
import random

print("UBIT NAME: puneethp")
print("Person Number: 50206906")

Train=55601
Vld=62612
valid = Vld - Train
Tst=69623
Test = Tst - Vld
data = np.genfromtxt("Querylevelnorm.txt",dtype=None,delimiter=None)
Y=np.zeros((Train,1),dtype='f8')
X=np.zeros((Train,46),dtype='float')
Yval=np.zeros((valid,1),dtype='f8')
Xval=np.zeros((valid,46),dtype='float')
#print("XValid size: ",Xvalid.shape)
#print("YValid size: ",Yvalid.shape)

Ytt=np.zeros((Test,1),dtype='f8')
Xtt=np.zeros((Test,46),dtype='float')

XC = np.zeros((Train,46),dtype='float')
YC = np.zeros((Train,1),dtype='f8')
X2Val = np.zeros((valid,46),dtype='float')
Y2Val = np.zeros((valid,1),dtype='f8')
X2Tt = np.zeros((Test,46),dtype='float')
Y2Tt = np.zeros((Test,1),dtype='f8')

M=4
W1=np.zeros((M,1),dtype='float')
WR1=np.zeros((M,1),dtype='float')

#Weight = np.zeros((1,4),dtype='float')

lamda=random.uniform(0,1)

def letor(X,Y,XV,YV,XT,YT,N1,N2,N3):
    
    phi1 = phi(X,Y,N1)
    WT = weight_closed(X,Y,phi1)
    ERMS = Ermscal(WT,X,Y,N1,phi1) 
    print("Weight of Closed form sol: ",WT)
    print("ERMS Closed form training set: ",ERMS)
       
    WTR = regweight_closed(X,Y,phi1)
    ERMSR = Ermscal(WTR,X,Y,N1,phi1)
    print("Weight of Reg Closed Form Sol: ",WTR)
    print("Regularized ERMS Closed form training set: ",ERMSR)
    
    phi2=phi(XV,YV,N2)
    ERMSV = Ermscal(WT,XV,YV,N2,phi2)
    print("ERMS Closed form validation set: ",ERMSV)
    
    RERMSV = Ermscal(WTR,XV,YV,N2,phi2)
    print("Regularized ERMS Closed form validation set: ",RERMSV)
    
    phi3 = phi(XT,YT,N3)
    ERMST = Ermscal(WT,XT,YT,N3,phi3)
    print("ERMS Closed form testing set: ",ERMST)
    
    RERMST = Ermscal(WTR,XT,YT,N3,phi3)
    print("Regularized ERMS Closed form testing set: ",RERMST)
    
    WTS = Stochastic(X,Y,phi1,N1,W1)
    SERMS = Ermscal(WTS,X,Y,N1,phi1)
    print("Weight of Stochastic: ",WTS)
    print("ERMS Stochastic form training set: ",SERMS)
    
    WTSR = Reg_Stochastic(X,Y,N1,phi1)
    SRERMS = Ermscal(WTSR,X,Y,N1,phi1)
    print("Weight of Stochastic Reg: ",WTSR)
    print("Regularized ERMS Stochastic from training set: ",SRERMS)
    
    phi4=phi(XV,YV,N2)
    SERMSV = Ermscal(WTS,XV,YV,N2,phi4)
    print("ERMS Stochastic form validation set: ",SERMSV)
    
    SRERMSV = Ermscal(WTSR,XV,YV,N2,phi4)
    print("Regularized ERMS Stochastic form validation set: ",SRERMSV)
    
    phi5 = phi(XT,YT,N3)
    SERMST = Ermscal(WTS,XT,YT,N3,phi5)
    print("ERMS Stochastic form testing set: ",SERMST)
    
    SRERMST = Ermscal(WTSR,XT,YT,N3,phi5)
    print("Regularized ERMS Stochastic form testing set: ",SRERMST)
    
    
    return

def phi(X,Y,N):
    
    sigma=np.zeros((46,46),dtype='float')
    for i in range(0,46):
        sigma[i][i] = np.var(X[:,i])
        #print(sigma)
        #
        #M=int(M)
       
    mu=np.zeros((M,46))
    #mu=1/46
    """
    for k in range(0,Train):
        for n in range(0,4):
            Z[k,n]=X[k,n]
            """
    #----------------------Generating Means----------------------
    for k in range(0,M):
        n=random.randint(0,N)
        mu[k]=X[n]
    #print("mu",len(mu[:,1]),len(mu[1,:]))
    #-----------------Calculating phi matrix----------------------
    phi=np.zeros((N,M),dtype='float')
    for i in range(0,N):
        for j in range(0,M):
            if(j==0):
                phi[i,j]=1
            else:
                t=(X[i]-mu[j])
                #print("t",t)
                t2=np.transpose(t)
                #print("t2",t2)
                t3=sigma.dot(t)
                #print("t3",t3)
                t4=t2.dot(t3)
                #print("t4",t4)
                #t4=t*t3
                t5=float(t4/2)
                phi[i,j]=(np.exp(-t5))
  #  print("phi: ",phi)
                
    return phi
    
def weight_closed(X,Y,phi):
    
    #------------------Calucalting Weights--------------
    W=np.zeros((M,1),dtype='float')
    p1=np.transpose(phi)
    p2=p1.dot(phi)
    p3=np.linalg.inv(p2)
    p4=p3.dot(p1)
    W=p4.dot(Y)
   # print("W",W)
    
    return W
    
def regweight_closed(X,Y,phi):
    
    #--------------------Calculating Regularized weights-------
    L=np.zeros((M,M),dtype="int")
    for i in range(0,M):
        L[i,i]=1
    p1=np.transpose(phi)
    p2=p1.dot(phi)
    L=lamda*L
    p3=L+p2
    p3=np.linalg.inv(p3)
    p4=p3.dot(p1)
    WR=p4.dot(Y)
   # print("WR",WR)
    
    return WR
    
def Ermscal(W,X,Y,N,phi):
    
    #--------------------Calculating EW--------------------
    #print("W:",W)
    w1=np.transpose(W)
    #print("W1 : ",w1)
  #  w2=w1.dot(W)
  #  EW=float(w2/2)
    #print("EW",EW)
    #--------------------Calculating ED--------------------
    h2=0
    #print("w1,phi[1]",w1,phi[1])
    for i in range(0,N):
        h=w1.dot(phi[i])
        #print("h: ",h)
        h2+=(Y[i]-h)*(Y[i]-h)  #----------EDW is being calculated--------
        #print("h2",h2)
    EDW=float(h2/2)
    #print("EDW",EDW)
    #--------------------Calculating ERMS-------------------
    ER1=(2*EDW)/N
    ERMS=np.sqrt(ER1)
  #  print("ERMS",ERMS)
    
    return ERMS
    
def Stochastic(X,Y,phi,N,W):
    
    W1 = np.zeros((M,1),dtype="float")
    W1[1] = [1]
    Erms1 = Ermscal(W1,X,Y,N,phi)
   # print("ERMS1: ",Erms1)
    ph1=np.zeros((M,1),dtype="float")
   # print("W1",W1)
    eta = 0.5
    for x in range(0,100):
     #   eta = 0.5 * eta
        ph = phi[x]
      #  print("ph",ph)
        for i in range(0,M):
            ph1[i]=ph[i]
        #print("Transpose",np.transpose(phi[x]))
        #ph1 = np.transpose(phi[x])
       # print("ph1",ph1)
        W2 = np.transpose(W1)
      #  print("pW2",W2)
        t2 = Y[x] - (W2.dot(ph1))
     #   print("t2",t2)
        t3 = - (ph1.dot(t2))
    #    print("t3: ",t3)
        
        dw = (eta*t3)
        de = dw / N
        W2 = W1 - de
    #    print("W2: ",W2)
        Erms2 = Ermscal(W2,X,Y,N,phi)
    #    print("ERMS2: ",Erms2)
       # if Erms2 - Erms1 <= 0.00001:
        #    break;
        Wt = W1
        W1 = W2
        
        
    return W2
    
    
def Reg_Stochastic(X,Y,N,phi):
    
    W1 = np.zeros((M,1),dtype="float")
    W1[1] = [1]
    ew = np.zeros((M,1),dtype="float")
    Erms1 = Ermscal(W1,X,Y,N,phi)
    #print("ERMS1: ",Erms1)
    ph1=np.zeros((M,1),dtype="float")
    #print("W1",W1)
    eta = 0.4
    for x in range(0,100):
     #   eta = 0.5 * eta
        ph = phi[x]
      #  print("ph",ph)
        for i in range(0,M):
            ph1[i]=ph[i]
        #print("Transpose",np.transpose(phi[x]))
        #ph1 = np.transpose(phi[x])
       # print("ph1",ph1)
        W2 = np.transpose(W1)
        #print("W2",W2)
        t2 = Y[x] - (W2.dot(ph1))
        #print("t2",t2)
        t3 = - (ph1.dot(t2))
        #print("t3: ",t3)
        #print("lamda: ",lamda)
        #de = dw / N
        ew = np.asarray(lamda)*W1
        #for i in W1:
         #   ew[i] = lamda*W1[i]
        #print("ew: ",ew)
        de = t3+ew
        #print("de: ",de)
        dw = (eta*de) / N
        #print("dw: ",dw)
        W2 = W1 - dw
      #  print("W2: ",W2)
    #    print("W2: ",W2)
        Erms2 = Ermscal(W2,X,Y,N,phi)
    #    print("ERMS2: ",Erms2)
       # if Erms2 - Erms1 <= 0.00001:
        #    break;
        Wt = W1
        W1 = W2
    #print("W2: ",W2)
    #print("ERMSREG2: ",Erms2)
   # print("ERMS REGULARized: ",Erms2)
   # print("ERMS REgularized Stochastic: ",Erms2)
        
    return W2
        
#------------------Obtaining X and Y matrix---------------------
for row in range(0,Train):
    rand=0
    for rand1 in range(0,48):
        if rand1==0:
            Y[row]=data[row][rand1]
        else:
            if rand1!=1:
                s=data[row][rand1]
                s=re.split('[:]',s)
                X[row][rand]=s[1]
                rand=rand+1

def initialize(XX,YY,N1,N2):      
    a=0
    for row in range(N1,N2):
        b=0
        c=0
        for d in range(0,48):
            if d==0:
                Ytt[c]=data[row][d]
                c=c+1
            else:
                if d!=1:
                    t=data[row][d]
                    t=re.split('[:]',t)
                    Xtt[a][b]=t[1]
                    b=b+1
        a=a+1
    return XX,YY
    
Xval,Yval = initialize(Xval,Yval,Train,Vld)
Xtt,Ytt = initialize(Xtt,Ytt,Vld,Tst)
#print("XVal,YVal: "<Xval,Yval)
#print("Xtt,Ytt: ",Xtt,Ytt)
#----------- CSV Data Training ------------

X1 = np.loadtxt(open("Querylevelnorm_X.csv","rb"),delimiter=",")
#print("CSV X: ",X1)

Y1 = np.loadtxt(open("Querylevelnorm_t.csv","rb"),delimiter=",")
#print("CSV Y: ",Y1)


print("Microsoft Real Time Data Set:")
letor(X,Y,Xval,Yval,Xtt,Ytt,Train,valid,Test)

for row in range(0,Train):
    for col in range(0,46):
        XC[row][col]=X1[row][col]
        
for row in range(0,Train):
    #print("Y1[row]",Y1[row])
    YC[row]=Y1[row]

def initializecsv(XD,YD,X1,Y1,N1,N2):    
    xs=0
    for row in range(N1, N2):
        for col in range(0,46):
            XD[xs][col]=X1[row][col]
        xs=xs+1

    ys=0
    for row in range(N1,N2):
        YD[ys]=Y1[row]
        
    return XD,YD
        
X2Val,Y2Val = initializecsv(X2Val,Y2Val,X1,Y1,Train,Vld)
X2Tt,Y2Tt = initializecsv(X2Tt,Y2Tt,X1,Y1,Vld,Tst)


print("CSV Data Set:")
letor(XC,YC,X2Val,Y2Val,X2Tt,Y2Tt,Train,valid,Test)


