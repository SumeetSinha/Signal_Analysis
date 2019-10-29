__author__ = "Sumeet K. Sinha"
__credits__ = [""]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Sumeet K. Sinha"
__email__ = "sumeet.kumar507@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy import integrate
import math


def FFT(x,dt,maxf,plot=True):

	"""
	Function to calculate FFT of a signal

	...

	Attributes
	----------
	x : float
		series data with equal spacing dt
	dt : float
		period of sampling data 
	maxf : int
		maximum frequency up to which FFT is desired
	plot : bool
		whether a plot should be showed or not (default True)

	Returns
	-------
	Freq : list
		frequency content in the signal
	Amp : list
		Amplitude for the corresponding frequencies
    """

	# Number of sample points
	N = x.size;
	# Total Time 
	T = N*dt;
	# sample spacing is dt
	# sampling frequency
	Fs = 1/dt;
	xfft  = scipy.fftpack.fft(x);
	xfreq = np.linspace(0.0, Fs/2, N/2);

	xfftHalf = 2.0/N * np.abs(xfft[:N//2]);
	xfftHalf[0] = xfftHalf[0]/2;

	if(plot):
		fig, ax = plt.subplots()
		ax.plot(xfreq, xfftHalf,'-k')
		if(maxf is not None):
			plt.xlim(0, maxf)
		plt.ylabel('Fourier transform |FFT(x)| [unit(x)]')
		plt.xlabel('Frequency [1/unit(dt)]')
		plt.show()

	Freq     = xfreq;
	Amp      = xfftHalf

	return Freq, Amp


def ResSpec(Acc,dt,damp,maxT,plot=True):

	"""
	Function to calculate Response Spectrum of an earthquake motion

	Attributes
	----------
	Acc : float
		acceleration series 
	dt : float
		period of sampling data 
	damp : float
		viscous damping in % 
	maxT : float
		maximum time period of evaluation 
	plot : bool
		whether a plot should be showed or not (default True)

	Returns
	-------
	T : list
		time period
	Sa : list
		maximum acceleration 
    """

	u  = 0*Acc;
	v  = 0*Acc;
	ac = 0*Acc;

	LengthAg = len(Acc);

	NumSteps = int(maxT/dt+1);
	T = np.linspace(0, maxT, num=NumSteps); # Time Period
	Sd = 0*T;	                     			 # Spectral Acceleration
	Sv = 0*T;                                    # Spectral Displacement
	Sa = 0*T;                                    # Spectral Acceleration


	for j in range(1,NumSteps):
		omega = 2.0*math.pi/T[j];
		m     = 1.0;                      # mass
		k     = omega*omega*m;            # stiffness
		c     = 2.0*m*omega*damp/100.0 # viscous damping
		K     = k+3.0*c/dt+6.0*m/(dt**2);
		a     = 6.0*m/dt+3.0*c;
		b     = 3.0*m+dt*c/2.0;

		# initial conditions 
		ac = 0*Acc;
		u  = 0*Acc;
		v  = 0*Acc;

		for i in range(0,LengthAg-1):
			df=-(Acc[i+1]-Acc[i])+a*v[i]+b*ac[i];  # delta Force
			du=df/K;
			dv=3.0*du/dt-3.0*v[i]-dt*ac[i]/2.0;
			dac=6.0*(du-dt*v[i])/(dt)**2.0-3.0*ac[i];
			u[i+1]=u[i]+du;
			v[i+1]=v[i]+dv;
			ac[i+1]=ac[i]+dac; 

		Sd[j]=np.amax( np.absolute(u));
		Sv[j]=np.amax( np.absolute(v));
		Sa[j]=np.amax( np.absolute(ac));

	Sa[0]=np.amax( np.absolute(Acc));

	if(plot):
		fig, ax = plt.subplots()
		ax.plot(T, Sa,'-k')
		plt.ylabel('Pseudo Response Acceleration (PSa) [unit(Acc)]')
		plt.xlabel('Time Period (T) [unit(dt)]')
		plt.show()

	return T, Sa


def Arias_Intensity(Acc,Time,plot=True):

	"""
	Function to calculate Arias intensity I
		I(t) = pi/(2g) (integral(0,t) Acc**2 dt)
		Ia = max(I)

	Attributes
	----------
	Acc : float
		acceleration series 
	Time : float
		time data or series 
	plot : bool
		whether a plot should be showed or not (default True)

	Returns
	-------
	Ia : float
		maximum Areas intensity 
	I : list
		cumulative Arias intensity with time as % of Ia
    """
	g  = 9.81;
	pi = math.pi;
	Acc = np.power(Acc,2);

	I  = pi/2/g*integrate.cumtrapz(y=Acc,x=Time,initial=0);
	Ia = max(I);

	I  = I/Ia*100;

	if(plot==True):
		fig, ax = plt.subplots()
		ax.plot(Time,I,'-k')
		plt.ylabel('Arias Intensity % ($I_a$ = '+str(round(Ia,4))+' m/s)')
		plt.xlabel('Time (T) [s]')
		plt.show()


	return Ia,I;