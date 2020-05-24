import codecs
with codecs.open('replaced_name.txt', 'r',' utf-8') as f:
    lines = f.readlines()

    linelist=[]
    for line in lines:

        line = line.split('\t')

        line[3] = line[3].replace(line[0],'实体A')
        line[3] = line[3].replace(line[1],'实体B')
        line[0] = '实体A'
        line[1] = '实体B'
        line = '\t'.join(line)
        linelist.append(line)
print(linelist)
linelist = ''.join(linelist)
print(linelist)
with codecs.open('nameE1E2infer.txt','w','utf-8') as f:
    f.write(linelist)
