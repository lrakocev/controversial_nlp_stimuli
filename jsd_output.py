import json 

with open('jsd_output.txt') as f: 
    data = f.read()

js = json.loads(data)