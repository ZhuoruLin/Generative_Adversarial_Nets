import numpy as np
import itertools
import os

file = []
for line in open('./list_attr_celeba.txt','r').read().splitlines():
    file.append(list(filter(None, line.split(' '))))
file.remove(file[0])
file = np.array(file)
columns = file[0]

legit = False
while not legit:
    attrs = np.unique(input('Input attrs to classify, split by space').split(' '))
    legit = True
    for attr in attrs:
        if attr not in file[0]:
            print(attr + 'is not an attribute')
            legit = False
values = []
for attr in attrs:
    element = []
    for j in [1, -1]:
        element.append(attr + '=' + str(j))
    values.append(element)
names = list(itertools.product(*values))
for name in names:
    directory = ''
    for i in range(len(name)):
        directory += (name[i])
        if i < len(name) - 1:
            directory += ' & '
    if not os.path.exists(directory):
        os.makedirs(directory)
for i in range(1, len(file)):
    classified_directory = './'
    for j in range(len(attrs)):
        attr = attrs[j]
        classified_directory += (attr + '=' + str(file[i][columns.index(attr) + 1]))
        if j < len(attrs) - 1:
            classified_directory += ' & '
    old_path = './img_align_celeba/' + file[i][0]
    new_path = classified_directory + '/' + file[i][0]
    os.rename(old_path, new_path)