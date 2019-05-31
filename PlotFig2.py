#Plots four files of data from AHP simulation

import csv
import numpy as np
import matplotlib.pyplot as plt

from numpy import mean
	

###############  Load Data

data1 = np.loadtxt("output1.dat")
data2 = np.loadtxt("output2.dat")
data3 = np.loadtxt("output3.dat")
data4 = np.loadtxt("output4.dat")

start = int(400/.025)
end = len(data1)

time1 = data1[start:end,0]
vm1 = data1[start:end,1]

time2 = data2[start:end,0]
vm2 = data2[start:end,1]

time3 = data3[start:end,0]
vm3 = data3[start:end,1]

time4 = data4[start:end,0]
vm4 = data4[start:end,1]

############### Spike interval function

def ISICalc(w):
	npoints = len(w)

	cc = 0
		
	stimes = [0.0]
	x = 0
	while x < npoints-1:
		if w[x] > 0:
			stimes.append(x*.025)
	
			cc += 1
			x += 300
		x += 1

	np = len(stimes)
	
	#first nISI
	fi = stimes[1] - stimes[0]
	nISI = [fi]
	x = 2
	while x < np-1:
		fi = stimes[x] - stimes[x-1]
		nISI.append(fi)
		x += 1
		
	a = nISI[np-5:np-1]

	return mn(a)		

def mn(w):				# mean function
	npoints = len(w)
	
	x = 0
	sum = 0
	while x < npoints:
		sum += w[x]
		x += 1
		
	return sum/npoints	
	
################## Calculate firing rates

a = ISICalc(vm1)
ISI = [a]

ISI.append(ISICalc(vm2))
ISI.append(ISICalc(vm3))
ISI.append(ISICalc(vm4))


############### AHP amplitude

def firstSpikeX(w):
	npoints = len(w)
	
	x = 0
	while w[x] < 0:
		x+=1
	return x
	
fAHP = [-55-vm1[firstSpikeX(vm1)+150]]
fAHP.append(-55-vm2[firstSpikeX(vm2)+150])
fAHP.append(-55-vm3[firstSpikeX(vm3)+150])
fAHP.append(-55-vm4[firstSpikeX(vm4)+150])


################## Plot traces
plt.figure(0)

plt.subplot(411)
plt.plot(time1, vm1,'k')	
plt.ylabel('Potential (mV)')
plt.xlabel('Time (ms)')


plt.subplot(412)
plt.plot(time2, vm2,'k')	
plt.ylabel('Potential (mV)')
plt.xlabel('Time (ms)')


plt.subplot(413)
plt.plot(time3, vm3,'k')	
plt.ylabel('Potential (mV)')
plt.xlabel('Time (ms)')


plt.subplot(414)
plt.plot(time4, vm4,'k')	
plt.ylabel('Potential (mV)')
plt.xlabel('Time (ms)')



###################### Plot ISI vs fAHP

## Regression

m,b = np.polyfit(fAHP,ISI,1)

fit = np.polyfit(fAHP,ISI,1)
fit_fn = np.poly1d(fit)

plt.figure(1)

plt.plot(fAHP, ISI, 'ko-')
plt.plot(fAHP, fit_fn(fAHP), 'r-')
plt.xlabel('fAHP (mV)')
plt.ylabel('ISI (ms)')
plt.show()




