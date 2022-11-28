'''

Para-Hockey 

'''

import streamlit as st
import numpy as np 
import pandas as pd
import scipy as sp
from scipy.signal import find_peaks
import glob
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import librosa as lib
from datetime import datetime
import timedelta
from plotly.subplots import make_subplots


st.image('hockeyCanada.png', width = 100)
#st.image('csi-pacific-logo-reverse.png', width = 100)
st.title("Para-Hockey Compound Sensor Analysis")


def lowpass(signal, highcut): 
	
	order = 4 

	nyq = 0.5*frequency
	highcut = highcut/nyq

	b,a = sp.signal.butter(order, [highcut], 'lowpass', analog=False)
	y = sp.signal.filtfilt(b,a, signal, axis = 0)
	return(y)

def highpass(signal, lowcut): 
	
	order = 4 

	nyq = 0.5*frequency
	lowcut = lowcut/nyq

	b,a = sp.signal.butter(order, [lowcut], 'highpass', analog=False)
	y = sp.signal.filtfilt(b,a, signal, axis = 0)
	return(y)



uploaded_data = st.text_input('Paste Folder Path')
date  = st.date_input('Input Data Collection Date')


if uploaded_data is not None: 
	files = glob.glob(f'{uploaded_data}*.txt')
	i4g = glob.glob(f'{uploaded_data}*.xlsx')[0]
	i4g = pd.read_excel(i4g)
	i4g['Date'] = pd.to_datetime(i4g['Date']).dt.date
	index = np.where(i4g['Date'] == date)[0]
	marker_types = i4g.groupby('Marker Name').count()


	marker_list = marker_types.index.tolist()
	IMU_start = np.where(i4g['Marker Name']=='Started IMU on Xsens App')[0][0]
	
	IMU_start_time = (i4g['Marker Hour'][IMU_start]*60*60) + (i4g['Marker Min'][IMU_start]*60) + i4g['Marker Sec'][IMU_start]
	

	
	global_time = (i4g['Marker Hour'][index]*60*60) + (i4g['Marker Min'][index]*60) + i4g['Marker Sec'][index] - IMU_start_time
	game_start = global_time[np.where(i4g['Marker Name']=='Puck Drop')[0][0]]
	num_periods = len(global_time[np.where(i4g['Marker Name']=='End of period')[0]])

	period_ends = []
	for i in range(num_periods):
		period_end = global_time[np.where(i4g['Marker Name']=='End of period')[0][i]]
		period_ends.append(period_end)


	game_end = global_time[np.where(i4g['Marker Name']=='End of period')[0][-1]]
	
	

	name_list = []
	for i in range(len(files)): 
		end = files[i].split('.')[0]
		start = end.split('-')[-1]
		name = start.split(' ')[1]
		name_list.append(name)

	col1, col2 = st.columns(2)
	with col1:
		collection = st.selectbox('Select Collection Type', 
			('Game', 'Practice'))
	
	with col2:
		last = st.selectbox('Select Player Name', 
			(name_list))


if uploaded_data is not None:
	uploaded_data_IMU = glob.glob(f'{uploaded_data}{collection}/{last}-Xsens DOT_*.csv')[0]
	time_csv = uploaded_data_IMU.split('_')[-1]
	IMU_start = time_csv.split('.')[0]
	IMU_start =  int(IMU_start)
	IMU_start = IMU_start +80000
	IMU_start = str(IMU_start)



	uploaded_dat_HR = glob.glob(f'{uploaded_data}{collection}/*{last}_samples.csv')[0]
		
	
	HR_data = uploaded_dat_HR
	HR_data = pd.read_csv(HR_data)
	HR = HR_data['HR [bpm]']
	peak_HR = HR.max()
	average_HR = HR.mean()
	
	time = HR_data['Time of day']
	
	strip_time = []

	for x in range(len(time)): 
		float_time = time[x].translate({ord(i): None for i in ':'})
		float_time = float_time.split('.')[0]
		strip_time.append(float_time)


	match = strip_time.index(IMU_start)

	HR = HR[match:].reset_index(drop=True)
	HR_peaks, _ = find_peaks(HR, height=peak_HR*.85, distance=1000)
	efforts = len(HR_peaks)
	
	strip_time = np.array(strip_time)
	
	HR_frequency = 10

	HR_time = []
	for i in range(len(HR_data[match:])):
		HR_time.append(i+1)

	HR_time = np.array(HR_time)/HR_frequency
	HR_peak_time = np.array(HR_time[HR_peaks])




	data = uploaded_data_IMU
	data = pd.read_csv(data)
	diff = data['SampleTimeFine'][1]-data['SampleTimeFine'][0]
	frequency = round((1/diff)*1000000)
	IMU_time = []
	for i in range(len(data)):
		IMU_time.append(i+1)

	IMU_time = np.array(IMU_time)/60

	

	accel_data = data[['Acc_X', 'Acc_Y', 'Acc_Z']]
	

	trans_accel = accel_data[['Acc_X', 'Acc_Y', 'Acc_Z']]

	
	trans_accel['Acc_X'] = lowpass(trans_accel['Acc_X'], 2)
	trans_accel['Acc_Y'] = lowpass(trans_accel['Acc_Y'], 2)
	trans_accel['Acc_Z'] = lowpass(trans_accel['Acc_Z'], 2)
	trans_accel['Acc_X'] = highpass(trans_accel['Acc_X'], 0.1)
	trans_accel['Acc_Y'] = highpass(trans_accel['Acc_Y'], 0.1)
	trans_accel['Acc_Z'] = highpass(trans_accel['Acc_Z'], 0.1)
	
	trans_accel = abs(trans_accel)

	rolling_sd = trans_accel.rolling(120).std()
	rolling_mean = trans_accel['Acc_Y'].rolling(20000).mean()-0.01
	zero_crossings = np.where(lib.zero_crossings(np.array(rolling_mean)))[0]

	
	period_start = np.array(zero_crossings[::2])
	period_start = np.delete(period_start, np.argwhere(np.ediff1d(period_start) <= 55000) + 1)
	period_end = np.array(zero_crossings[1::2])-20000

	
	res_trans_accel = ((trans_accel['Acc_Z'])**2 + (trans_accel['Acc_Y'])**2)**(.5)
	max_res_accel = res_trans_accel.max()
	moving_accel = res_trans_accel[np.where(res_trans_accel >= 0.5)[0]]
	average_accel = moving_accel.mean()
	active_time = len(moving_accel)/frequency
	
	fig = make_subplots(specs=[[{"secondary_y": True}]])
	fig.add_trace(go.Scatter(x=IMU_time, y=res_trans_accel,
                mode='lines',
                name='Resultatnt Acceleration'
                ))
	
	fig.add_vline(x=game_start, line_width=1, line_dash="dash", line_color=  '#2ca02c' , annotation_text= 'Game Start', annotation_textangle = 270, annotation_position="top left")
	
	
	fig.add_vline(x=game_end, line_width=1, line_dash="dash", line_color=  '#e377c2' , annotation_text= 'Game End', annotation_textangle = 270, annotation_position="top left")

	fig.add_trace(go.Scatter(x=HR_peak_time, y=HR[HR_peaks],
                mode='markers',
                marker=dict(
		            color='Red',
		            size=10),
                name='HR Peaks'
                ),secondary_y=True,)
	fig.add_trace(go.Scatter(x=HR_time, y=HR,
                mode='markers',
                marker=dict(
		            color='LightSkyBlue',
		            size=3),
                name='Heart Rate [BPM]'
                ),secondary_y=True,)
		
	for i in range(len(global_time)):
		fig.add_vline(x=global_time[i], line_width=1, line_dash="dash", line_color=  '#7f7f7f' , annotation_text= i4g['Marker Name'][i], annotation_textangle = 90)

	fig.update_layout(title = "Para-Hockey In Game Sensor Analysis", 
							xaxis_title = '<b>Time</b> (s)')

	fig.update_yaxes(title_text="<b>Acceleration</b> (m/s^2)", secondary_y=False)
	fig.update_yaxes(title_text="<b>Beats Per Minute</b> (BPM)", secondary_y=True, showgrid = False)
	fig.update_layout(xaxis=dict(showgrid=False),
              yaxis=dict(showgrid=False))
	st.plotly_chart(fig)

	Sensor_outputs = pd.Series([date, collection, str(datetime.strptime(IMU_start, '%H%M%S')).split(' ')[-1], str(datetime.strptime(IMU_start, '%H%M%S')+timedelta.Timedelta(seconds=int(IMU_time[-1]))).split(' ')[-1], efforts, max_res_accel, peak_HR, average_accel, average_HR, active_time/60])

	export_data = pd.DataFrame([list(Sensor_outputs)], columns = ['Event Date','Activity type', 'Block Start', 'Block End', 'Number of Efforts', 'Max Acceleration', 'Max Heart Rate', 'Average Acceleration', 'Average Heart Rate', 'Active Minutes'])

	export_data.index = [last]

	st.write(export_data)



	






