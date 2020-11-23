import json 
import ast

file = open("jsd_output.txt", "r")
contents = file.read()
js = ast.literal_eval(contents)
