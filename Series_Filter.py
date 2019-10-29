import math;
import numpy as np; 
from scipy import special;
from scipy import signal
from scipy.interpolate import UnivariateSpline;



def Butterworth_Filter(x,order,Low_High,Wn):

    """
    Function to apply low and high pass Butterworth filter on series data

    ...

    Attributes
    ----------
    x : float
        series data with equal spacing dt 
    order : int
        order of the Butterworth filter
    Low_High : string
        'low': applies low pass filter 
        'high': applies high pass filter 
    Wn: int (0,1)
        normalized frequency with respect to Nyquist frequency 
        0 : filter with 0 freq  
        1 : filter with Nyquist frequency 

    Returns
    -------
    y : list
        filtered signal x
    """


    # Create an order butterworth filter:

    b, a = signal.butter(order,Wn,Low_High);

    # Apply the filter to x. Use lfilter_zi to choose the initial condition of the filter:

    zi   = signal.lfilter_zi(b, a);
    z, _ = signal.lfilter(b, a, x, zi=zi*x[0]);

    # Apply the filter again, to have a result filtered at an order the same as filtfilt:

    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

    # Use filtfilt to apply the filter:

    y = signal.filtfilt(b, a, x);

    return y;
	

def Reject_Outliers_With_Mean(x, m = 3.):

    """
    Reject data in x whose values are greater than 'm' times standard 
    deviation from the global mean  

    ...

    Attributes
    ----------
    x : float
        series data
    m : float
        number of standard deviation from mean

    Returns
    -------
    x : list
        filtered signal x with nan in place of outliers *default:3)
    """

    N = len(x);
    mean = np.mean(x);
    std  = np.std(x);

    mdev = np.abs(x-mean)/std;

    for i in range(0,N):
        if(mdev[i]>m):
            x[i]=np.nan;

    return x;

def Reject_Outliers_With_Median(x, m = 3.):

    """
    Reject data in x whose values are greater than 'm' times 
    median deviation from the global median (MAD)  

    ...

    Attributes
    ----------
    x : float
        series data
    m : float
        number of deviation from MAD (default:3)

    Returns
    -------
    x : list
        filtered signal x with nan in place of outliers 
    """

    N = len(x);
    median = np.median(x);

    dev = np.abs(x-median)
    MAD = -1/(math.sqrt(2)*special.erfcinv(1.5))*np.median(dev);
    mdev = dev/MAD
    # print(MAD)

    for i in range(0,N):
        if(mdev[i]>m):
            x[i]=np.NaN;

    return x;

def Reject_Outliers_With_Mov_Median(x, window_size=None, m = 3.):
    """
    Reject data in x whose values are greater than 'm' times median deviation 
    from the median (MAD) within a windows size of W

    ...

    Attributes
    ----------
    x : float
        series data
    window_size: int
        number id data data points for the window (default: len(x))
    m : float
        number of deviation from MAD (default:3)

    Returns
    -------
    x : list
        filtered signal x with nan in place of outliers 
    """
    N = len(x);

    if(window_size==None):
        window_size = N*2+1;

    W        = window_size;
    New_Data = np.zeros(N);

    W = int(math.floor(W/2)*2+1);

    for i in range(0,N):

        New_Data[i] = x[i];

        lower_range= max(0,i-int((W-1)/2));
        upper_range= min(N-1,i+int((W-1)/2));

        window_data = x[lower_range:upper_range+1];

        median = np.median(window_data);
        dev    = np.abs(window_data-median)
        MAD    = -1/(math.sqrt(2)*special.erfcinv(1.5))*np.median(dev);
        mdev   = abs(x[i]-median)/MAD

        if(mdev>m):
            New_Data[i]=np.nan;

    x = New_Data/1.0;

    return x;

def Reject_and_Fill_Outliers(data, m = 3.):
    import numpy as np
    import pandas as pd
    from scipy import stats
       
    data_shape  = data.shape;    
    index       = np.array(range(0,data_shape[0]));
    
    if(len(data_shape)==1):
        data = np.reshape(data, (data_shape[0], 1));
              
    new_data    = data*0;
    data_shape  = data.shape;
        
    num_cols    = data.shape[1];
       
    for i in range(0,num_cols):
        
        column_data = data[:,i];
        df = pd.DataFrame({'Values': column_data});      
        df = df[(np.abs((df.values - df.values.mean())/(df.values.std()+1e-3)) < m).all(axis=1)];
        
        i_ = df.index;
        d_ = df.values[:,0];
                   
        data_r = np.interp(index, i_, d_);
        new_data[:,i] = data_r

    x = new_data/1.0;
           
    return x;


def Offset_Data(x, offset):

    """
    offset the whole data by a scalar 

    ...

    Attributes
    ----------
    x : float
        series data
    offset: float
        offset scalar value

    Returns
    -------
    x : list
        data with offset 
    """

    import numpy as np;
    
    offset = np.ones(x.shape)*offset;
    x = x-offset;
    return x;


def Mov_Mean(data,window_size):
    N = len(data);
    W = window_size;
    New_Data = data*0;

    W = int(math.floor(W/2)*2+1);

    for i in range(0,N):

        lower_range= max(0,i-int((W-1)/2));
        upper_range= min(N-1,i+int((W-1)/2));

        window_data = data[lower_range:upper_range+1];
        # print(window_data)
        mean        = np.mean(window_data);
        New_Data[i] = mean;

    return New_Data

def Mov_Median(data,window_size):
    N = len(data);
    W = window_size;
    New_Data = data*0;

    W = int(math.floor(W/2)*2+1);

    for i in range(0,N):

        lower_range= max(0,i-int((W-1)/2));
        upper_range= min(N-1,i+int((W-1)/2));

        window_data = data[lower_range:upper_range+1];
        # print(window_data)
        mean        = np.median(window_data);
        New_Data[i] = mean;

    return New_Data

def Mov_Mean_With_Weight(data,window_size,weight):
    N = len(data);
    W = window_size;
    New_Data = data*0;

    W = int(math.floor(W/2)*2+1);

    for i in range(0,N):

        lower_range= max(0,i-int((W-1)/2));
        upper_range= min(N-1,i+int((W-1)/2));

        window_data = data[lower_range:upper_range+1];
        window_weight = weight[lower_range:upper_range+1];
        # print(window_data)
        mean        = np.mean(np.multiply(window_data,window_weight))/np.mean(window_weight);
        New_Data[i] = mean;

    return New_Data

def Reject_data(data,threshold=0.001):
    N = len(data);
    New_Data = np.copy(data);

    for i in range(1,N):

        if(New_Data[i]>=(min(New_Data[:i]+threshold))):
            New_Data[i]=np.nan;

    return New_Data

def Interpolate_With_Smooth_Spline(data,smooth=0.85):
    N = len(data);
    x = np.linspace(1,N,N);
    y = data;
    w = np.isnan(y);

    s = UnivariateSpline(x[~w], y[~w]);

    s.set_smoothing_factor(smooth)

    return s(x)

