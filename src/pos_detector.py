import numpy as np
from tqdm import tqdm

from constants import *


def e_step(paradigms, abs_forms):
    pos_distr = {pos:ALPHA for pos in range(POS_N)}
    abs_form_distr = {pos:{f:ALPHA for f in abs_forms} for pos in range(POS_N)}
    
    for p in paradigms:
        for pos in range(POS_N):
            pos_distr[pos] += float(p["distr"][pos])
            for f in p["filtered_abs_par"]:
                if f in abs_form_distr[pos]:
                    abs_form_distr[pos][f] += float(p["distr"][pos])

    tot = sum(pos_distr.values())
    for pos in range(POS_N):
        pos_distr[pos] /= tot
        pos_tot = sum(abs_form_distr[pos].values())
        for abs_form in abs_form_distr[pos]:
            if abs_form in abs_form_distr[pos]:
                abs_form_distr[pos][abs_form] /= pos_tot
    return pos_distr, abs_form_distr


def m_step(paradigms, pos_distr, abs_form_distr):
    for p in paradigms:
        new_distr = np.array([0.0 for i in range(POS_N)])
        for pos in range(POS_N):
            prob = pos_distr[pos]
            for abs_form in p["filtered_abs_par"]:
                if abs_form == "X0":
                    continue
                if abs_form in abs_form_distr[pos]:
                    prob *= abs_form_distr[pos][abs_form]
            new_distr[pos] = prob
        if new_distr.sum() > 0:
            p["distr"] = new_distr / new_distr.sum()
        else:
            p["distr"] = new_distr


def get_distr():
    distr = None
    distr = np.zeros((POS_N,1))
    distr[randint(0,POS_N-1)] = 1
    return (distr / distr.sum())


def identify_pos(data,abs_forms):
    print("Identify POS (running EM)")
    for p in data:
        p["distr"] = get_distr()

    pos_distr, abs_form_distr = None, None
    for i in tqdm(range(1000)):
        pos_distr, abs_form_distr = e_step(data, abs_forms)
        m_step(data, pos_distr, abs_form_distr)
    print("")

    poses = [[] for i in range(POS_N)]

    for p in data:
        pos = poses[p["distr"].argmax()]
        pos.append(p)

    for i,pos in enumerate(poses):
        print("POS",i,":",len(pos),"lexemes")
        print("Sample paradigms:")
        for p in sample(pos,min(10,len(pos))):
            print(" ".join(p["filtered_tokens"])," ".join(p["filtered_abs_par"]))
        print("Most frequent abstract forms:")
        print(Counter([f for p in pos for f in p["filtered_abs_par"]]).most_common(5))
        print()                
    return poses