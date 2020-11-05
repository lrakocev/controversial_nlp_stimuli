import os

import glob
file_list = glob.glob('fs*.out')

fail_list = []
for file in file_list:
    with open(file, 'rb') as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        last_line = f.readline().decode()

        if last_line.strip() != "freeview.bin: cannot connect to X server":
        	fail_list.append(file)


for file in fail_list:
	fp = open(file)
	for i, line in enumerate(fp):
	    if i == 9:
	        print(line)
	    elif i > 9:
	        break
	fp.close()