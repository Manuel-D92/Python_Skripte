
from functools import partial
import numpy as np
import sympy as sy
import math
import matplotlib.pyplot as plt


## Points there are adjusted
#def punkte():
#    x = [2.1,2.6,3.8,4.5,5.8,6.0]
#    y = [0.3,1.2,1.6,2.7,3.2,3.8]
#    return x,y

## Points there are adjusted
def punkte():
    x = [1,1.1,2,3,4,6]
    y = [-3,-3,0,2,0,-17]
    return x,y

def Start_value():
    x_0 = [8,1,2,4]
    return x_0

#def model():
#    x= []; y=[];a = []; b=[]
#    #a,b = punkte()
#    m = (a*x+b) -y
#    return m

def F(func,x_0,x,y,x1,y1,var):
    f = []
    for i in range(len(x)):
        er = {x1: x[i], y1: y[i]}
        for ii in range(len(var)):
            er[var[ii]] = x_0[ii]
        f.append(-func.evalf(subs=er))
    return f

def F_d(func_div,x_0,x,y,x1,y1,var):
    f_d=[]
    for i in range(len(x)):
        f_zw = []
        for j in range(len(func_div)):
            func_zw = func_div[j]
            er = {x1: x[i], y1: y[i]}
            for ii in range(len(var)):
                er[var[ii]] = x_0[ii]
            f_zw.append(func_zw.evalf(subs=er))
        f_d.append(f_zw)
    return f_d

def F_1(func,x_0,x,y,x1,y1,var):
    f = []
    for i in range(len(x)):
        er = {x1:x[i],y1:y[i]}
        for ii in range(len(var)):
            er[var[ii]]=x_0[ii]
        f.append(func.evalf(subs=er)) # {x1:x[i],y1:y[i]}    ,var[0]:x_0[0],var[1]:x_0[1],var[2]:x_0[2]
    return f

def linA_vektor(f,f_d,mu):
    len_eye = len(f_d[0])
    for i in range(len_eye):
        f.append(0)
    eye = np.eye(len_eye,len_eye,k=0,dtype=int)*mu
    for j in range(len(eye)):
        zwi = []
        for k in range(len(eye)):
            zwi.append(eye[j,k])
        f_d.append(zwi)
    return f,f_d


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    iteration = 12

    ## Weights
    a = sy.Symbol("a")
    b = sy.Symbol("b")
    c = sy.Symbol("c")
    d = sy.Symbol("d")


    x1= sy.Symbol('x')
    y1 = sy.Symbol("y")
    var = []
    var.append(a)
    var.append(b)
    var.append(c)
    var.append(d)

    ### fitting-function
    #func = sy.sin(y1*a)*sy.cos(x1*b) + sy.sin(c)*sy.cos(x1*d) #+sy.sin(y1)*sy.cos(y1*b)
    #func = y1*a - x1*sy.exp(b*x1) - y1*1.8226577 - x1*sy.exp(0.00836199*x1) + y1*2.23747892956970 - x1*sy.exp(0.0459126795235036*x1)
    func =  ((x1*a) + sy.exp(b*(y1)))
    #func = ((x1 - a) ** 2 + sy.exp(b * (x1 ** 2 + y1 ** 2)) - 5)
    #func = ((a*x1+b) -y1) * ((c*x1+d)-y1)
    #func = a*x1**3 +c*x1**2 +d*x1**1 +b+y1
    #func = ((b+a*x1+c*x1)*d+(2+1*x1+3*x1)*1)-y1   #NN
    #func = y1 + c*(x1-a)*(x1-b)

    func_div=[]
    for v in range(len(var)):
        func_div.append(sy.diff(func,var[v]))

    x,y = punkte()
    x_0 = Start_value()
#    model = model()
    f = F(func,x_0,x,y,x1,y1,var)
    #f = F(x_0,x,y)
    f_d = F_d(func_div,x_0,x,y,x1,y1,var)
    mu = 1
    status = -1
    #for k in range(4):
    k=0
    while(k!=iteration):
        if(status<0):
            f_eye,f_d_eye = linA_vektor(f,f_d,mu)
            print(f)
            print(f_d)
            L,r = np.linalg.qr(np.array(f_d_eye))
            p = np.dot(L.T, f_eye)
            s_k = np.dot(np.linalg.inv(r),p)
            x_1 = x_0+s_k
            print(L)
            print(s_k)
            print(x_1)
            f_1 = F_1(func,x_1,x,y,x1,y1,var)
            print(f_1)
            f = F(func,x_0,x,y,x1,y1,var)
            f_d = F_d(func_div,x_0,x,y,x1,y1,var)
            F_F_h = np.array(f)*-1 + (np.array(f_d).dot(s_k))
            print(F_F_h)

            roh_mu = (np.linalg.norm(np.array(f,dtype=float)) ** 2 - np.linalg.norm(np.array(f_1,dtype=float)) ** 2) / (np.linalg.norm(np.array(f,dtype=float)) ** 2 - np.linalg.norm(np.array(F_F_h,dtype=float)) ** 2)
            #roh_mu = (np.linalg.norm(f) ** 2 - np.linalg.norm(f_1) ** 2) / (
            #            np.linalg.norm(f) ** 2 - np.linalg.norm(F_F_h) ** 2)

            print(roh_mu)
            status = roh_mu
            if (status < 0):
                mu = mu *2
        if(status>0):
            mu = mu / 2
            x_0 = x_1
            f = F(func, x_0, x, y, x1, y1,var)
            # f = F(x_0,x,y)
            f_d = F_d(func_div, x_0, x, y, x1, y1,var)

            f_eye, f_d_eye = linA_vektor(f, f_d, mu)
            print(f)
            print(f_d)
            L, r = np.linalg.qr(np.array(f_d_eye))
            p = np.dot(L.T, f_eye)
            s_k = np.dot(np.linalg.inv(r), p)
            x_1 = x_0 + s_k
            print(L)
            print(s_k)
            print(x_1)
            f_1 = F_1(func, x_1, x, y, x1, y1,var)
            print(f_1)
            f = F(func, x_0, x, y, x1, y1,var)
            f_d = F_d(func_div, x_0, x, y, x1, y1,var)
            F_F_h = np.array(f) * -1 + (np.array(f_d).dot(s_k))
            print(F_F_h)
            roh_mu = (np.linalg.norm(np.array(f,dtype=float)) ** 2 - np.linalg.norm(np.array(f_1,dtype=float)) ** 2) / (np.linalg.norm(np.array(f,dtype=float)) ** 2 - np.linalg.norm(np.array(F_F_h,dtype=float)) ** 2)

            print(roh_mu)
            status = roh_mu

        print("all right", k)
        k=k+1


    plt.plot(x,y,"o")


    print("umstellen..")
    y_umgestellt = sy.solve(func,y1)
    y_umgestellt = y_umgestellt[0]
    print("fertig umstellen")

    x_value= np.linspace(0,10,50)
    y_value=[]
    y_pn_value=[]


    for ypn in range(len(x)):
        er = {x1: x[ypn]}
        for ii in range(len(var)):
            er[var[ii]] = x_0[ii]
        y_pn_value.append(sy.re(y_umgestellt.evalf(subs=er)))

    for yn in range(len(x_value)):
        er = {x1: x_value[yn]}
        for ii in range(len(var)):
            er[var[ii]] = x_0[ii]
        y_value.append(sy.re(y_umgestellt.evalf(subs=er)))

    y_test = np.array(y_value)
    plt.plot(x_value,np.array(y_value,dtype=float))
    plt.plot(x,np.array(y_pn_value,dtype=float),"rx")


    print("\na = ",x_1[0],"\nb = ", x_1[1],"\nc = ", x_1[2])
    mse = np.sqrt(np.sum((y-np.array(y_pn_value,dtype=float))**2))
    print("MSE: " , mse)
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
