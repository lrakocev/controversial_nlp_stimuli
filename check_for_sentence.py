import os
import re

import glob
file_list = glob.glob('sample_judge*.out')

success_id_list = []
for file in file_list:
    f = open(file)
    x = f.read()
    if "The advice would be greatly great" in x:
        print(file)

    f.close()