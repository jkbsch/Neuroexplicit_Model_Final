import numpy as np
#source: https://www.pythonpool.com/viterbi-algorithm-python/ 05th Oct 2023  now adapted
"""observations = ("normal", "cold", "dizzy") # brauchen wir auch nicht
states = ("Healthy", "Fever")
start_p = {"Healthy": 0.6, "Fever": 0.4}
trans_p = {
    "Healthy": {"Healthy": 0.7, "Fever": 0.3},
    "Fever": {"Healthy": 0.4, "Fever": 0.6},
}
emit_p = {
    "Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
    "Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6},
}""" # wir brauchen emit nicht - wir haben gleich die Wkt der einzelnen States

states = np.array(["Healthy", "Fever"]) # hier also: Healthy = 0, Fever = 1
start_p = np.array([0.6, 0.4]) #Wkt Healthy im Start = 0.6, Wkt Fever im Start = 0.4
trans_p = np.array([[0.7, 0.3], [0.4, 0.6]]) #Bsp: Wkt von Healthy zu Healthy = 0.7, von Healthy zu Fever = 0.3
probs_states = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]) #Bsp: Wkt durch unsere beobachteten Daten, dass am Anfang State = Healthy liegt bei 0.4;  Wkt zu Zeit t (3 Zeitpunkte) im State i zu sein (2 States)

def viterbi_algorithm(states, start_p, trans_p,probs_states):
    V = np.ones((len(probs_states), len(probs_states[0]), 2)) * (-1) #[[Healthy:[prob, prev], Fever:[prob, prev]], Zeitpunkt 2]: hier wird jeweils die aktuelle Wkt und der zug. Vorgänger gespeichert
    print(start_p * probs_states[0])
    print(V)
    for st in range(len(states)): #Berechne Wkt dafür, zu Zeitpunkt 0 in State st zu landen
        print(st)
        V[0] = [start_p[st] * probs_states[0][st], None]
        print(V)

    for t in range(1, len(observations)): #jetzt für alle zukünftigen Zeitpunkte
        #print(V)
        V.append({})
        for st in states:
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
            prev_st_selected = states[0] # zuerst für st 0, dann für die restlichen: berechne den vorherigen State, der am wahrscheinlichsten der Vorgängerstate war
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p[st][observations[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected} # berechne mit wahrscheinlichstem Vorgängerstate die Gesamtwkt, im aktuellen State zu sein


    for line in dptable(V):
        print(line)

    opt = []
    max_prob = 0.0
    best_st = None
    print("V:", V, "\n \n V.item",V[-1].items() )

    for st, data in V[-1].items(): #erstelle den besten Pfad: beginne beim letzten und gehe dann durch
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print("The steps of states are " + " ".join(opt) + " with highest probability of %s" % max_prob)


def dptable(V):
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

viterbi_algorithm(states, start_p, trans_p, probs_states)