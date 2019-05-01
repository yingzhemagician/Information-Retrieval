import numpy as np
import re
import math

correct = 0
incorrect = 0

doc_path = "tank.tokenized"

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
            doc_vector.append(new_doc_vec)
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
        else:
            if temp_list[i] in new_doc_vec.keys():
                weight = new_doc_vec.get(temp_list[i])
                new_doc_vec.setdefault(temp_list[i], weight + 1)
            else:
                new_doc_vec.setdefault(temp_list[i], 1)

V_term = []
V_sum1 = []
V_sum2 = []
num_1 = 0
num_2 = 0

print("V_profile calculating...")
for doc_index in range(1, 4001):

    if sensenum[doc_index] == '1':
        num_1 += 1
    elif sensenum[doc_index] == '2':
        num_2 += 1

    for term, weight in doc_vector[doc_index].items():

        if sensenum[doc_index] == '1':
            if term in V_term:
                i = V_term.index(term)
                V_sum1[i] += weight

            else:
                V_term.append(term)
                V_sum1.append(1)
                V_sum2.append(0)

        elif sensenum[doc_index] == '2':
            if term in V_term:
                i = V_term.index(term)
                V_sum2[i] += weight

            else:
                V_term.append(term)
                V_sum2.append(1)
                V_sum1.append(0)

#print("sum1:", num_1, " sum2:", num_2)

LLike = [0 for n in range(len(V_sum2))]

for i in range(len(V_sum2)):
    if V_sum1[i] > 0:
        LLike[i] = math.log(V_sum1[i] / (V_sum2[i] + 1))
    else:
        LLike[i] = math.log(0.2/(V_sum2[i] + 1))

test = '0'
j = 0
sumLLike = 0
for test_index in range(1, 401):
    V_test = [0 for n in range(len(V_term))]
    for term, value in test_vector[test_index].items():
        #if term in V_term:
        j = V_term.index(term)
        V_test[j] += 1

    for i in range(len(V_test)):
        sumLLike += LLike[i] * V_test[i]

    if sumLLike > 0:
        test = '1'
    elif sumLLike < 0:
        test = '2'
    else:
        test = '0'

    if sensenum[3600 + test_index] == test:
        correct += 1
        print(test, "    #", test_index)
    else:
        incorrect += 1
        print(test, "*   #", test_index)

    sumLLike = 0

print("correct = ", correct)
print("incorrect = ", incorrect)
print("accuracy = %.2f%%"  %(correct/(correct + incorrect)*100))