"""
wget --user chec --password chec2048 https://www.mpi-hd.mpg.de/personalhomes/white/checs/data/d2018-05-14_DynamicRange_noNSB_5degC_gainmatched-200mV/Run43461_r0.tio
NEEDS SORTING. CURRENTLY NOT ABLE TO OUTPUT CAMERA DATA.
"""
import sys
sys.path.append('../')
sys.path.append('../../')
from CHECLabPy.core.io import TIOReader
import os
import numpy as np
#%matplotlib inline
from matplotlib import pyplot as plt
from CHECLabPy.plotting.camera import CameraImage
input_dir= "/Users/chec/Desktop/CHECdata/410181000NSB0NSB0PE"#gaindropatdifferentpe"
ievent = 10  # We will view the 10th event

input_path = os.path.join(input_dir, "data_Run108_r1.tio")  # The path to the r1 run file
reader = TIOReader(input_path)  # Load the file
wfs = reader[ievent]  # Obtain the waveforms for the 10th event     #print("This event contains the samples for {} pixels and {} samples".format(wfs.shape[0], wfs.shape[1]))
plt.plot(wfs[20], label='1000MHz,0 PE, pixel 20')
input_path = os.path.join(input_dir, "data_Run028_r1.tio")  # The path to the r1 run file
reader = TIOReader(input_path)  # Load the file
wfs = reader[ievent]  # Obtain the waveforms for the 10th event
plt.plot(wfs[20], label='40MHz,0 PE, pixel 20')
plt.xlabel("Time (ns)")
plt.ylabel("Signal (mV)")


readalldatain=1
plotdata=1
plotfinal=1
plotr1oncam=1
if readalldatain==1:
    allevents=np.zeros((64,128,878))
    input_path = os.path.join(input_dir, "data_Run108_r1.tio")  # The path to the r1 run file
    reader = TIOReader(input_path)  # Load the file
    for i in range(0,878):
        ievent = i  # We will view the 10th event
        wfs = reader[ievent]  # Obtain the waveforms for the 10th event     #print("This event contains the samples for {} pixels and {} samples".format(wfs.shape[0], wfs.shape[1]))
        for j in range (0,64):
            allevents[j,:,i]=wfs[j]
if plotdata==1:
    a=sum(allevents)/64
    a=np.transpose(a)
    b=sum(a)/878
    plt.plot(b,label='1000MHz,0PE, average over 11196 events, 64 pixels')
plt.legend(loc='top left')
plt.ylim ((-1,2))
plt.show()

'''
input_path = os.path.join(input_dir, "data_Run00559_r1.tio")  # The path to the r1 run file
reader = TIOReader(input_path)  # Load the file
wfs = reader[ievent]  # Obtain the waveforms for the 10th event     #print("This event contains the samples for {} pixels and {} samples".format(wfs.shape[0], wfs.shape[1]))
plt.plot(wfs.mean(0), label='1000MHz,low PE, average')
input_path = os.path.join(input_dir, "data_Run00319_r1.tio")  # The path to the r1 run file
reader = TIOReader(input_path)  # Load the file
wfs = reader[ievent]  # Obtain the waveforms for the 10th event
plt.plot(wfs.mean(0), label='0MHz,low PE, average')
plt.legend(loc='top left')
plt.show()

input_path = os.path.join(input_dir, "data_Run00533_r1.tio")  # The path to the r1 run file
reader = TIOReader(input_path)  # Load the file #print("This run contains {} events, {} pixels, and {} samples".format(reader.n_events, reader.n_pixels, reader.n_samples))
wfs = reader[ievent]  # Obtain the waveforms for the 10th event #print("This event contains the samples for {} pixels and {} samples".format(wfs.shape[0], wfs.shape[1]))
plt.plot(wfs.mean(0), label='1000MHz,100PE,average')
input_path = os.path.join(input_dir, "data_Run00293_r1.tio")  # The path to the r1 run file
reader = TIOReader(input_path)  # Load the file
wfs = reader[ievent]  # Obtain the waveforms for the 10th event
plt.plot(wfs.mean(0), label='0MHz,100PE,average')
plt.legend(loc='top left')
plt.show()

input_path = os.path.join(input_dir, "data_Run00265_r1.tio")  # The path to the r1 run file
reader = TIOReader(input_path)  # Load the file
wfs = reader[ievent]  # Obtain the waveforms for the 10th event
plt.plot(wfs.mean(0), label='0MHz,2500 PE, average')
input_path = os.path.join(input_dir, "data_Run00505_r1.tio")  # The path to the r1 run file
reader = TIOReader(input_path)  # Load the file
wfs = reader[ievent]  # Obtain the waveforms for the 10th event
plt.plot(wfs.mean(0), label='1000MHz,2500 PE, average')
plt.legend(loc='top left')
plt.show()
plt.close()

#plt.rcParams['figure.figsize'] = [15, 7]
#lines = plt.plot(wfs[200])  # Plot the waveform for pixel 200 for the 10th event
#plt.show()

# lines = plt.plot(wfs.T)  # Plot the waveform for all pixels for the 10th event
# lines = plt.plot(wfs.mean(0))  # Plot the average waveform across all pixels for the 10th event
#simple_dead_pixels = np.where(wfs[:, 60] < 10)[0]
#print("A simple estimate of the number of dead pixels = {}".format(len(simple_dead_pixels)))
#dead_pixel_mask = np.zeros(reader.n_pixels, dtype=np.bool)
#dead_pixel_mask[simple_dead_pixels] = True
#lines = plt.plot(wfs[~dead_pixel_mask].T)  # Plot the waveform for all pixels for the 10th event (except dead pixels)
#lines = plt.plot(wfs[:, 60])  # Plot value of the samples at 60ns against pixel

# Generate a CameraImage object using the classmethod "from_mapping" which accepts the
# mapping object contained in the reader, which converts from pixel ID to pixel
# coordinates (using the Mapping class in TargetCalib)
camera=plt.figure(figsize=(10,10))
camera = CameraImage.from_mapping(reader.mapping)
camera.add_colorbar()
camera.image = wfs[:, 60]  # Plot value of the sample at 60ns for each pixel
#camera.highlight_pixels(simple_dead_pixels, color='red', linewidth=3)  # Highlight the dead pixels
print('a')
plt.show()
'''