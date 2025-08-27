"""
Dynamic TS-SESCA: Tri-Subpopulation Sigmoid-Enhanced SCA
  • Subpop-1 (Mixed) : 50 % （随机 sin/cos，a，r3∼U(0,2)）
  • Subpop-2 (Sin)   : 25 % （sin更新，放大步长 1.5a，r3∼U(1,3)）
  • Subpop-3 (Cos)   : 25 % （cos更新，缩小步长 0.5a，r3∼U(0,1)）
每轮根据适应度重新分配个体，更新后记录全局最优。
"""

import os
import numpy as np
from copy import deepcopy
from opfunu.cec_based.cec2022 import *

# === 参数设置 ===
PopSizeTotal = 100
DimSize      = 100
TrialRuns    = 20
MaxFEs       = 1000 * DimSize
curIter      = 0
FuncNum      = 1
MaxIter      = int(MaxFEs / PopSizeTotal * 2)

Pop  = np.zeros((PopSizeTotal, DimSize))
Fit  = np.zeros(PopSizeTotal)
LB   = None
UB   = None

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def Check(ind, LB, UB):
    for k in range(len(ind)):
        if ind[k] > UB[k]:
            ind[k] = UB[k] - (ind[k] - UB[k])
        elif ind[k] < LB[k]:
            ind[k] = LB[k] + (LB[k] - ind[k])
    return ind

def Initialization(func, LB, UB):
    global Pop, Fit
    for i in range(PopSizeTotal):
        Pop[i] = LB + (UB - LB) * np.random.rand(len(LB))
        Fit[i] = func(Pop[i])

def Update_Subpop(indices, pbest, a, LB, UB, func, mode):
    for i in indices:
        old, ofit = Pop[i].copy(), Fit[i]
        new = old.copy()
        for j in range(len(old)):
            if mode == 'sin':
                base = np.sin(2*np.pi*np.random.rand())
                step = 1.5 * a
                r3 = np.random.uniform(1, 3)
            elif mode == 'cos':
                base = np.cos(2*np.pi*np.random.rand())
                step = 0.5 * a
                r3 = np.random.uniform(0, 1)
            else:
                base = (np.sin if np.random.rand()<0.5 else np.cos)(2*np.pi*np.random.rand())
                step = a
                r3 = np.random.uniform(0, 2)
            factor = 2 * sigmoid(base) - 1
            new[j] = old[j] + step * factor * abs(r3 * pbest[j] - old[j])
        new = Check(new, LB, UB)
        nf  = func(new)
        if nf < ofit:
            Pop[i], Fit[i] = new, nf

def TSSESCA_operator(func, LB, UB):
    global Pop, Fit, curIter, MaxIter
    a = 2 - curIter * (2 / MaxIter)
    sorted_idx = np.argsort(Fit)
    n25 = PopSizeTotal // 4
    Cos_idx   = sorted_idx[:n25]
    Sin_idx   = sorted_idx[-n25:]
    Mixed_idx = sorted_idx[n25:-n25]
    gbest = Pop[np.argmin(Fit)]

    Update_Subpop(Mixed_idx, gbest, a, LB, UB, func, 'mixed')
    Update_Subpop(Sin_idx,   gbest, a, LB, UB, func, 'sin')
    Update_Subpop(Cos_idx,   gbest, a, LB, UB, func, 'cos')

def Run_TSSESCA(func, LB, UB):
    global curIter, TrialRuns, MaxIter
    all_runs = []
    for run in range(TrialRuns):
        np.random.seed(2025 + 25*run)
        Initialization(func, LB, UB)
        curIter = 0
        best_curve = [np.min(Fit)]
        while curIter < MaxIter:
            TSSESCA_operator(func, LB, UB)
            curIter += 1
            best_curve.append(np.min(Fit))
        all_runs.append(best_curve)
    np.savetxt(f"./TS-SESCA/TS-SESCA_Data/CEC2022/F{FuncNum+1}_{len(LB)}D.csv",
               all_runs, delimiter=",")

def main(dim):
    global DimSize, MaxFEs, MaxIter, Pop, Fit, LB, UB, FuncNum
    DimSize = dim
    MaxFEs  = dim * 1000
    MaxIter = int(MaxFEs / PopSizeTotal * 2)
    LB = np.full(dim, -100.0)
    UB = np.full(dim,  100.0)
    Pop = np.zeros((PopSizeTotal, dim))
    Fit = np.zeros(PopSizeTotal)

    CEC2022 = [F12022(dim), F22022(dim), F32022(dim), F42022(dim),
               F52022(dim), F62022(dim), F72022(dim), F82022(dim),
               F92022(dim), F102022(dim), F112022(dim), F122022(dim)]
    for i, f in enumerate(CEC2022):
        FuncNum = i
        Run_TSSESCA(f.evaluate, LB, UB)

if __name__ == "__main__":
    if not os.path.exists("./TS-SESCA/TS-SESCA_Data/CEC2022"):
        os.makedirs("./TS-SESCA/TS-SESCA_Data/CEC2022")
    for Dim in [10, 20]:
        main(Dim)
