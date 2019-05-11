file = open("ascii/words.txt","r")
lines = file.readlines()
file.close()
file = open("words_label.txt","w")
for line in lines:
    if line.startswith("#"):
        continue
    file.write(line)
file.close()
