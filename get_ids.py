


fp = open("lines.txt")
for i, line in enumerate(fp):
	if i%2 == 0:
		relevant = line.split("/")
		print(relevant[6][3:])

