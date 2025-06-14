from scipy.io import loadmat
import numpy as np
import math
from scipy import signal
from scipy.signal import cheb1ord, filtfilt, cheby1
from numpy.linalg import inv, eig
from scipy.io import loadmat
import h5py

Fs = 1000. # sampling freq

################################################################################
# prepare training or test dataset
# input:
#       subj: subject index
#       runs: trial index
#       tw: time window length
#       cl: # of total frequency classes
#       overlapping_rate: non-overlapping rate of two adjacent windows
#       permutations: channel indexes
#       Fs: sampling rate
# output:
#       x: dataset [?,tw,ch], ?=cl*runs*samples
#       y: labels [?,1]
#       all_freqs: true frequencies
#
def prepare_data_as(subj,runs,tw,idx,
    cl=15,non_overlapping_rate=0.25,permutation=[0,1,2,3,4,5,6,7,8,9],Fs=1000.):

    all_freqs = loadmat('/home/devin/data_MAC/SSVEP Dataset/DJL Dataset/Freq_Phase.mat')['freqs'][0] # true freq

    step = int(math.ceil(tw*non_overlapping_rate)) # step of overlapping window
    ch = len(permutation) # # of channels
    x = np.array([],dtype=np.float32).reshape(0,tw,ch) # data
    y = np.zeros([0],dtype=np.int32) # true label

    # file = loadmat('/home/devin/data_MAC/SSVEP Dataset/DJL Dataset/'+'S'+str(subj)+'_ssvep.mat')['eeg']
    with h5py.File('/home/devin/data_MAC/SSVEP Dataset/DJL Dataset/' + f'S{subj}_ssvep.mat', 'r') as f:
        file_ssvep = np.array(f['eeg'])     # file.shape：(50, 14, 10, 1640)

    with h5py.File('/home/devin/data_MAC/SSVEP Dataset/DJL Dataset/' + f'S{subj}_idle.mat', 'r') as f:
        file_idle_all = np.array(f['eeg'])     # file.shape：(840, 1, 10, 1640)
    
    file_idle = file_idle_all[idx]                                      # (50, 1, 10, 1640)

    # 得到 (50, 15, 10, 1640) —— 14 个频率 + 1 个 idle
    file = np.concatenate([file_ssvep, file_idle], axis=1)    

    file = file.transpose(2, 3, 1, 0)   # (trial, freq, ch, len) → (ch, len, freq, trial)
    for run_idx in runs:
        for freq_idx in range(cl):
            raw_data = file[permutation,640:1640,freq_idx,run_idx].T
            n_samples = int(math.floor((raw_data.shape[0]-tw)/step))
            _x = np.zeros([n_samples,tw,ch],dtype=np.float32)
            _y = np.ones([n_samples],dtype=np.int32) * freq_idx
            for i in range(n_samples):
                _x[i,:,:] = raw_data[i*step:i*step+tw,:]

            x = np.append(x,_x,axis=0) # [?,tw,ch], ?=runs*cl*samples
            y = np.append(y,_y)        # [?,1]

    # x = filter(x)

    print('S'+str(subj)+'|x',x.shape)
    return x, y, all_freqs

################################################################################
# prepare reference signals
# input:
#       subj: subject index
#       runs: trial index
#       tw: time window length
#       cl: # of total frequency classes
#       overlapping_rate: non-overlapping rate of two adjacent windows
#       permutations: channel indexes
#       Fs: sampling rate
# output:
#       template: reference signal [?,cl,tw,ch], ?=cl*samples
#
def prepare_template(subj,runs,tw,idx,
    cl=15,non_overlapping_rate=.25,permutation=[0,1,2,3,4,5,6,7,8,9],Fs=1000.):

    step = int(math.ceil(tw*non_overlapping_rate)) # step of overlapping window
    ch = len(permutation) # # of channels
    tr = len(runs)
    n_samples = int(math.floor((1640-640-tw)/step))

    template = np.zeros([n_samples,cl,tw,ch],dtype=np.float32)
    weight = np.zeros([n_samples,cl,ch],dtype=np.float32)

    with h5py.File('/home/devin/data_MAC/SSVEP Dataset/DJL Dataset/' + f'S{subj}_ssvep.mat', 'r') as f:
        file_ssvep = np.array(f['eeg'])     # file.shape：(50, 14, 10, 1640)

    with h5py.File('/home/devin/data_MAC/SSVEP Dataset/DJL Dataset/' + f'S{subj}_idle.mat', 'r') as f:
        file_idle_all = np.array(f['eeg'])     # file.shape：(840, 1, 10, 1640)
    
    file_idle = file_idle_all[idx]                                      # (50, 1, 10, 1640)

    # 得到 (50, 15, 10, 1640) —— 14 个频率 + 1 个 idle
    file = np.concatenate([file_ssvep, file_idle], axis=1)    
    file = file.transpose(2, 3, 1, 0)   # (trial, freq, ch, len) → (ch, len, freq, trial)

    # file = loadmat('/home/devin/data_MAC/SSVEP Dataset/DJL Dataset/'+'S'+str(subj)+'.mat')['data']
    for freq_idx in range(cl):
        raw_data = np.zeros([ch,1640-640,tr],dtype=np.float32)
        for r in range(tr):
            raw_data[:,:,r] = file[permutation,640:1640,freq_idx,runs[r]] # [ch,1250,runs]

        # build template
        for i in range(n_samples):
            _t = np.zeros([ch,tw,tr],dtype=np.float32)
            for r in range(tr):
                _t[:,:,r] = raw_data[:,i*step:i*step+tw,r]
            # _t = filter(_t) # filter 7 - 70 Hz
            template[i,freq_idx,:,:] = np.mean(_t,axis=-1).T

    template = np.tile(template, (cl,1,1,1)) # [cl*sample,cl,tw,ch]
    print('S'+str(subj)+'|template',template.shape)


    return template # [cl,samples]

def prepare_ref(subj,runs,tw,
    cl=15,non_overlapping_rate=.25,permutation=[0,1,2,3,4,5,6,7,8,9],Fs=1000.):


    step = int(math.ceil(tw*non_overlapping_rate)) # step of overlapping window
    ch = 10 # # of channels
    tr = len(runs)
    n_samples = int(math.floor((1640-640-tw)/step))

    template = np.zeros([n_samples,cl,tw,ch],dtype=np.float32)
    # weight = np.zeros([n_samples,cl,ch],dtype=np.float32)


    # file = loadmat('/home/devin/data_MAC/DeepLearning/code_transfer/branch_fintune/Benchmark/'+'S'+str(subj)+'.mat')['data']
    # file 64*1500*40*6
    # template 1880*40*150*9
    # generate sin-cos reference signals
    fs = 1000
    TP = np.arange(0, 1, 1 / fs)  # Time points
    file_path = r'/home/devin/data_MAC/SSVEP Dataset/DJL Dataset/Freq_Phase.mat'
    data = loadmat(file_path)
    frequencies = data['freqs'].flatten()
    phases = data['phases'].flatten()
    num_of_harmonics = 5

    file_ref = np.zeros([10,1000,15],dtype=np.float32)

    for id_label in range(14):
        f = frequencies[id_label].item()
        phase = phases[id_label].item()
        ref_signal = []
        for h in range(1, num_of_harmonics + 1):
            sinh = np.sin(2 * np.pi * h * f * TP + h * phase)
            cosh = np.cos(2 * np.pi * h * f * TP + h * phase)
            ref_signal.extend([sinh, cosh])
        file_ref[:,:,id_label] = ref_signal

    for freq_idx in range(cl):
        raw_data = np.zeros([ch,1640-640],dtype=np.float32)
        raw_data[:,:] = file_ref[permutation,:,freq_idx] # [ch,1250,runs]

        # build template
        for i in range(n_samples):
            _t = np.zeros([ch,tw],dtype=np.float32)
            _t[:,:] = raw_data[:,i*step:i*step+tw]

            template[i,freq_idx,:,:] = _t.T # np.mean(_t,axis=-1).T # 如果runs = 1,则会出错

    template = np.tile(template, (cl,1,1,1)) # [cl*sample,cl,tw,ch]
    print('S'+str(subj)+'|ref',template.shape)
    

    return template # [cl,samples]

def normalize(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            _x = x[i,:,j]
            x[i,:,j] = (_x - _x.mean())/_x.std(ddof=1)
    return x



## prepossing by Chebyshev Type I filter
def filter(x):
    nyq = 0.5 * Fs
    Wp = [6/nyq, 90/nyq];
    Ws = [4/nyq, 100/nyq];
    N, Wn=cheb1ord(Wp, Ws, 3, 40);
    b, a = cheby1(N, 0.5, Wn,'bandpass');
    # --------------
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            _x = x[i,:,j]
            x[i,:,j] = filtfilt(b,a,_x,padlen=3*(max(len(b),len(a))-1)) # apply filter

    return x
