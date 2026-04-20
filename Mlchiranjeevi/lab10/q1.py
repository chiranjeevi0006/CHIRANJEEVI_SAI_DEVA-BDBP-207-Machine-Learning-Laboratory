import math

def entropy(labels):
    yes = labels.count("Yes")
    no = labels.count("No")
    total = yes + no

    if total == 0:
        return 0

    p_yes = yes / total
    p_no = no / total

    ent = 0
    if p_yes > 0:
        ent -= p_yes * math.log2(p_yes)
    if p_no > 0:
        ent -= p_no * math.log2(p_no)

    return ent

data = {
        "sunny": ["No", "No", "No"],
        "rainy": ["Yes", "Yes", "No"],
        "overcast": ["Yes", "Yes"]
    }

all_labels=[]
for i in data.values():
    all_labels.extend(i)

    print("Entropy:", entropy(all_labels))

def information_gain(data):
    total_labels = []
    for i in data.values():
        total_labels.extend(i)

    total_entropy = entropy(total_labels)

    weighted_entropy = 0
    total_len = len(total_labels)

    for i in data.values():
        weight = len(i) / total_len
        weighted_entropy += weight * entropy(i)

    return total_entropy - weighted_entropy


print("Information Gain:", information_gain(data))