import matplotlib.pyplot as plt
import csv
import sys	

x = []
d = []
a = []
o = []
g = []

with open(sys.argv[1],'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
    	print row[0]
    	if(float(row[0])%100==0):
	        x.append(float(row[0]))
	        d.append(float(row[1]))
	        a.append(float(row[2])/100)
	        o.append(float(row[3])/100)
	        g.append(float(row[4]))

plt.plot(x,d, label='D Loss')
plt.plot(x,a, label='Accuracy')
plt.plot(x,o, label='Op. Accuracy')
plt.plot(x,g, label='G Loss')
plt.xlim([0.0, 20000])
plt.ylim([0.0, 3.0])
plt.xlabel('Epochs')
plt.ylabel('')
plt.title('Classical AC GAN')
plt.legend()
plt.savefig('PlotImages/classacgan.png')
plt.show()