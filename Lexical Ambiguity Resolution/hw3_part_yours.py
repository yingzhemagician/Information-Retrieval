import numpy as np
import re
import math

correct = 0
incorrect = 0


def cosine_sim_a(vec1, vec2, vec1_norm = 0.0, vec2_norm = 0.0):
    if not vec1_norm:
        vec1_norm = sum(v * v for v in vec1)
    if not vec2_norm:
        vec2_norm = sum(v * v for v in vec2)

    # save some time of iterating over the shorter vec
    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1

    # calculate the cross product
    cross_product = np.dot(vec1, vec2)
    return cross_product / math.sqrt(vec1_norm * vec2_norm)


doc_path = "plant.tokenized"###########

doc_vector = [[]]
test_vector = [[]]
sensenum = [[]]
doc_num = 0

common_words = set()

for line in open("common_words", 'r'):
    if line:
        common_words.add(line.strip())

temp_list = []
print(doc_path, " inputting...")
for word in open(doc_path, 'r'):
    word = word.strip()

    if word[:2] == ".I":
        if temp_list:
            x_index = temp_list.index(".x-")############
            for i in range(len(temp_list)):
                if x_index == i:
                    pass
                elif x_index - i in [1, -1]:
                    if x_index - i == 1:
                        temp_list[i] = "L-" + temp_list[i]
                    elif x_index -i == -1:
                        temp_list[i] = "R-" + temp_list[i]

                    if temp_list[i] in new_doc_vec.keys():
                        weight = new_doc_vec.get(temp_list[i])
                        new_doc_vec.setdefault(temp_list[i], weight + 6)
                    else:
                        new_doc_vec.setdefault(temp_list[i], 6)

                elif x_index - i in [2, -2, 3, -3]:
                    if temp_list[i] in new_doc_vec.keys():
                        weight = new_doc_vec.get(temp_list[i])
                        new_doc_vec.setdefault(temp_list[i], weight + 4)
                    else:
                        new_doc_vec.setdefault(temp_list[i], 4)

                elif x_index - i in [4, -4]:
                    if temp_list[i] in new_doc_vec.keys():
                        weight = new_doc_vec.get(temp_list[i])
                        new_doc_vec.setdefault(temp_list[i], weight + 2)
                    else:
                        new_doc_vec.setdefault(temp_list[i], 2)

                else:
                    if temp_list[i] in new_doc_vec.keys():
                        weight = new_doc_vec.get(temp_list[i])
                        new_doc_vec.setdefault(temp_list[i], weight + 1)
                    else:
                        new_doc_vec.setdefault(temp_list[i], 1)


        doc_num += 1
        new_doc_vec = {}

        sensenum.append(word[-1])

        if doc_num <= 3600:
            doc_vector.append(new_doc_vec)

        else:
            test_vector.append(new_doc_vec)

        temp_list = []

    elif word in common_words or not re.search("[a-zA-Z]", word):
        pass
    else:
        if word[:3] == ".x-":
            temp_list.append(".x-")
        else:
            temp_list.append(word)

if temp_list:
    x_index = temp_list.index(".x-")  ############
    for i in range(len(temp_list)):
        if x_index == i:
            pass
        elif x_index - i in [1, -1]:
            if x_index - i == 1:
                temp_list[i] = "L-" + temp_list[i]
            elif x_index - i == -1:
                temp_list[i] = "R-" + temp_list[i]


            if temp_list[i] in new_doc_vec.keys():
                weight = new_doc_vec.get(temp_list[i])
                new_doc_vec.setdefault(temp_list[i], weight + 6)
            else:
                new_doc_vec.setdefault(temp_list[i], 6)

        elif x_index - i in [2, -2, 3, -3]:
            if temp_list[i] in new_doc_vec.keys():
                weight = new_doc_vec.get(temp_list[i])
                new_doc_vec.setdefault(temp_list[i], weight + 3)
            else:
                new_doc_vec.setdefault(temp_list[i], 3)

        elif x_index - i in [4, -4]:
            if temp_list[i] in new_doc_vec.keys():
                weight = new_doc_vec.get(temp_list[i])
                new_doc_vec.setdefault(temp_list[i], weight + 2)
            else:
                new_doc_vec.setdefault(temp_list[i], 2)
        else:
            if temp_list[i] in new_doc_vec.keys():
                weight = new_doc_vec.get(temp_list[i])
                new_doc_vec.setdefault(temp_list[i], weight + 1)
            else:
                new_doc_vec.setdefault(temp_list[i], 1)

V_term = []
V_profile1 = []
V_profile2 = []
num_1 = 0
num_2 = 0

print("V_profile calculating...")
for doc_index in range(1, 3601):

    if sensenum[doc_index] == '1':
        num_1 += 1
    elif sensenum[doc_index] == '2':
        num_2 += 1

    for term, weight in doc_vector[doc_index].items():

        if sensenum[doc_index] == '1':
            if term in V_term:
                i = V_term.index(term)
                V_profile1[i] += weight

            else:
                V_term.append(term)
                V_profile1.append(1)
                V_profile2.append(0)

        elif sensenum[doc_index] == '2':
            if term in V_term:
                i = V_term.index(term)
                V_profile2[i] += weight

            else:
                V_term.append(term)
                V_profile2.append(1)
                V_profile1.append(0)

print("1:", num_1, " 2:", num_2)

for i in range(len(V_profile1)):
    V_profile1[i] = V_profile1[i] / num_1

for i in range(len(V_profile2)):
    V_profile2[i] = V_profile2[i] / num_2

j = 0
sim1 = 0
sim2 = 0
test = '0'
for test_index in range(1, 401):
    V_test = [0 for n in range(len(V_term))]
    for term, value in test_vector[test_index].items():
        if term in V_term:
            j = V_term.index(term)
            V_test[j] += 1

    sim1 = cosine_sim_a(V_profile1, V_test)
    sim2 = cosine_sim_a(V_profile2, V_test)

    if sim1 > sim2:
        test = '1'
    elif sim2 > sim1:
        test = '2'
    else:
        test = '0'

    if sensenum[3600 + test_index] == test:
        correct += 1
        print(test, "    #", test_index, "sim1-sim2 = ", sim1 - sim2)
    else:
        incorrect += 1
        print(test, "*   #", test_index, "sim1-sim2 = ", sim1 - sim2)

print("correct = ", correct)
print("incorrect = ", incorrect)
print("accuracy = %.2f%%"  %(correct/(correct + incorrect)*100))