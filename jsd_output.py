import json 

file = open("jsd_output.txt", "r")
contents = file.read()
js = ast.literal_eval(contents
