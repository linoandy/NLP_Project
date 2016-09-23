import csv

def writetoCSV(newrow, filename):
	filename = filename + ".csv"
	c = csv.writer(open(filename, "a"))
	c.writerow(newrow)

l = [('','Result_1', 'Result_2', 'Result_3', 'Result_4'), ('auto', 1, 2, 3, 4), ('space', 5, 6, 7, 8)]
l = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
l = zip(*l)
for item in l:
	writetoCSV(item, 'hehehe')
