#source: https://www.pythonpool.com/viterbi-algorithm-python/ 05th Oct 2023
observations = ("normal", "cold", "dizzy")
states = ("Healthy", "Fever")
start_p = {"Healthy": 0.6, "Fever": 0.4}
trans_p = {
    "Healthy": {"Healthy": 0.7, "Fever": 0.3},
    "Fever": {"Healthy": 0.4, "Fever": 0.6},
}
emit_p = {
    "Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
    "Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6},
}

def viterbi_algorithm(observations, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states: #Berechne Wkt dafür, zu Zeitpunkt 0 in State st zu landen
        V[0][st] = {"prob": start_p[st] * emit_p[st][observations[0]], "prev": None}
        print(V)

    for t in range(1, len(observations)): #jetzt für alle zukünftigen Zeitpunkte
        #print(V)
        V.append({})
        for st in states:
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
            prev_st_selected = states[0] # zuerst für st 0, dann für die restlichen: berechne den vorherigen State, der Wkt maximiert, um zum aktuellen state zu kommen
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p[st][observations[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}


    for line in dptable(V):
        print(line)

    opt = []
    max_prob = 0.0
    best_st = None

    for st, data in V[-1].items():
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

viterbi_algorithm(observations, states, start_p, trans_p, emit_p)