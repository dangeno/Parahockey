'''

Para-Hockey 

'''

import streamlit as st
import os
import numpy as np 
import pandas as pd
import scipy as sp
from scipy.signal import find_peaks
import glob
import plotly.graph_objects as go
import librosa as lib
from datetime import datetime
import timedelta
from plotly.subplots import make_subplots
import scipy.integrate as integrate
from scipy.signal import savgol_filter



st.image('hockeyCanada.png', width = 100)
#st.image('csi-pacific-logo-reverse.png', width = 100)
st.title("Para-Hockey Compound Sensor Analysis")


def lowpass(signal, highcut, frequency): 
	
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

def detect_strokes(data): 
	stroke_peaks, _ = find_peaks(data, height= 2.5, distance=frequency/2)

	return(stroke_peaks)


def group_data(array, threshold):
	    groups = []
	    current_group = [array[0]]

	    for i in range(1, len(array)):
	        diff = array[i] - array[i - 1]
	        if diff > threshold:
	            groups.append(current_group)
	            current_group = [array[i]]
	        else:
	            current_group.append(array[i])
	    if len(current_group) >= 200:  # Append the last group if it has at least 200 items
        	groups.append(current_group)

	    return groups



uploaded_data = st.text_input('Paste Folder Path', value="Please Insert Path")

if uploaded_data == 'Please Insert Path': 
	st.header('Upload Data')
	st.stop()


else:

	folders = glob.glob(f'{uploaded_data}*/', recursive = True)
	
	folder_list = []
	
	for f in folders:
		folder = f.split('/')[-2]
		folder_list.append(folder)

	
	game = st.selectbox('Select Game for Analysis', folder_list)

	IMU_files = glob.glob(f'{uploaded_data}/{game}/Xsens*.csv')
	IMU_num_list = []
	
	for file in IMU_files: 
		IMU_number = file.split('/')[-1]
		IMU_number = IMU_number.split(' ')[1]
		IMU_number = IMU_number.split('_')[0]
		IMU_num_list.append(IMU_number)


		time_csv = file.split('_')[-1]
		IMU_on = time_csv.split('.')[0]
		IMU_on =  int(IMU_on)


	files = glob.glob(f'{uploaded_data}/{game}/*.txt')
	name_list = []


	for name in files: 
		name_first = name.split('_')[5]
		name_last = name.split('_')[6]
		name_full = f'{name_first} {name_last}'
		name_list.append(name_full)

	markers = st.checkbox('Show Game Markers')
	process_all = st.checkbox('Process All Game Data')
	if process_all == True: 
		athlete_sel = name_list
	else: 
		athlete_sel = st.multiselect('Select a Player to analyze', name_list, default=name_list[0])
	
	
	athlete_export_app = []
	
	for player in athlete_sel:
		try:
			fig = make_subplots(specs=[[{"secondary_y": True}]])
			name_first = player.split(' ')[0]
			name_last = player.split(' ')[1]

			player_HR = glob.glob(f'{uploaded_data}/{game}/*{name_last}*.csv')[0]
			
			player_IMU_num = player_HR.split('/')[10]
			player_IMU_num = int(player_IMU_num.split('_')[2])

			
			player_HR_data = pd.read_csv(player_HR)
			
			polar_log = glob.glob(f'{uploaded_data}*.xlsx')
			polar_log_path = polar_log[0]
			polar_log = pd.read_excel(polar_log_path)
				
			IMU_on = str(IMU_on)

			date = datetime(int(game.split('-')[0]), int(game.split('-')[1]), int(game.split('-')[2]))
			date = date.date()

			
			polar_log['Date'] = pd.to_datetime(polar_log['Date']).dt.date
			index = np.where(polar_log['Date'] == date)[0]
			polar_log = polar_log[index[0]:(index[-1]+1)]
			marker_types = polar_log.groupby('Marker Name').count()
			marker_list = marker_types.index.tolist()



			IMU_start_frame = np.where(polar_log['Marker Name'].str.contains("Start") & polar_log['Marker Name'].str.contains("IMU"))
			IMU_start_frame = IMU_start_frame[0]+index[0]
			
			
			
			IMU_start = []
			if len(IMU_start_frame) == 1: 
				IMU_start = int(IMU_start_frame)
			if len(IMU_start_frame) == 2:

				num_range_1 = polar_log['Marker Name'][IMU_start_frame[0]].split(' ')[2]
				num_start_1 = int(num_range_1.split('-')[0])
				num_end_1 = int(num_range_1.split('-')[1])
				

				num_range_2 = polar_log['Marker Name'][IMU_start_frame[1]].split(' ')[2]
				num_start_2 = int(num_range_2.split('-')[0])
				num_end_2 = int(num_range_2.split('-')[1])

				if num_start_1<=player_IMU_num<=num_end_1:
					IMU_start = IMU_start_frame[0]
					
				elif num_start_2<=player_IMU_num<=num_end_2:
					IMU_start = IMU_start_frame[1]
					
				else: 
					st.write('nope')
			else: 
				st.write('cannot find start')
				st.stop()
				

			

			IMU_start_time = (polar_log['Marker Hour'][IMU_start]*60*60) + (polar_log['Marker Min'][IMU_start]*60) + polar_log['Marker Sec'][IMU_start]
			

			
			global_time = (polar_log['Marker Hour'][index]*60*60) + (polar_log['Marker Min'][index]*60) + polar_log['Marker Sec'][index] - IMU_start_time


			
			
			game_start = global_time[np.where(polar_log['Marker Name']=='Puck Drop')[0][0]+index[0]]


			#finding start and end of periods
			period_start_masks = [polar_log['Marker Name'].str.contains(keyword, case=False) for keyword in ['period', 'start']]
			combined_mask = period_start_masks[0] & period_start_masks[1]
			period_starts = polar_log[combined_mask].index

			period_ends = polar_log[polar_log['Marker Name'].str.contains('End of period', case=False)].index

			
			num_periods = len(period_starts)

			

			puck_drops = np.where(polar_log['Marker Name']=='Puck Drop')[0]+index[0]
			drop_times = global_time[puck_drops]

			play_end = global_time[puck_drops+1]
			#st.write(play_end)

			#play_length = np.array(play_end) - np.array(drop_times)


			game_end = global_time[np.where(polar_log['Marker Name']=='End of period')[0][-1]+index[0]]

			
			HR_data = player_HR_data
			
			HR = HR_data['HR [bpm]']
			peak_HR = HR.max()
			average_HR = HR.mean()
			
			time = HR_data['Time of day']
			
			strip_time = []

			for x in range(len(time)): 
				float_time = time[x].translate({ord(i): None for i in ':'})
				float_time = float_time.split('.')[0]
				strip_time.append(float_time)


			match = strip_time.index(IMU_on)

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

			IMU_data = glob.glob(f'{uploaded_data}/{game}/Xsens {player_IMU_num}*.csv')
			IMU_data = IMU_data[0]
			
			data = pd.read_csv(IMU_data,skiprows=1, usecols=[i for i in range(15)], dtype=float)

			
			diff = data['SampleTimeFine'][1]-data['SampleTimeFine'][0]
			frequency = round((1/diff)*1000000)
			IMU_time = []
			

			for i in range(len(data)):
				IMU_time.append(i+1)

			IMU_time = np.array(IMU_time)/frequency



			accel_data = data[['Acc_X', 'Acc_Y', 'Acc_Z']]
			

			trans_accel = accel_data[['Acc_X', 'Acc_Y']]
			

			rolling_sd = trans_accel['Acc_X'].rolling(500).std()
			rolling_sd = savgol_filter(rolling_sd,int(len(accel_data)/100),2)
			Threshold_sd = np.where(rolling_sd >= 1)[0]
			IMU_game_end = np.where(IMU_time >= game_end)[0][0]
			Threshold_sd = Threshold_sd[Threshold_sd <= IMU_game_end]

			
			shift_indexes = group_data(Threshold_sd,2100)
			shift_indexes = [shift for shift in shift_indexes if len(shift) >= 1200]
			shift_number = len(shift_indexes)

			lateral_accel = trans_accel['Acc_Y']
			res_trans_accel = trans_accel['Acc_X']
			window = 200
			trace_accel = abs(res_trans_accel).rolling(window).mean()


			shift_length = []
			shift_strokes = []
			stroke_shift_num = []

			#change to frequency

			for shift in shift_indexes: 
				shift_length.append((shift[-1]-shift[0])/60)
				strokes = detect_strokes(res_trans_accel[shift])+shift[0]
				stroke_shift_num.append(len(strokes))
				shift_strokes.append(strokes)

			all_strokes = np.concatenate(shift_strokes)

			rolling_mean = trans_accel['Acc_Y'].rolling(20000).mean()-0.01

			zero_crossings = np.where(lib.zero_crossings(np.array(rolling_mean)))[0]


			
			stroke_time =  np.array(IMU_time[all_strokes])


			res_trans_accel = lowpass(res_trans_accel, 15,60)
			res_trans_accel = highpass(res_trans_accel, 0.1)

			max_res_accel = res_trans_accel.max()
			moving_accel = res_trans_accel[np.where(res_trans_accel >= 0.5)[0]]
			average_accel = moving_accel.mean()
			active_time = len(moving_accel)/frequency

			shift_onset = []
			shift_offset = []

			puck_drops = list(puck_drops)
			play_end = list(play_end)
			trace_accel = trace_accel[window:]
			trace_accel = trace_accel - np.mean(trace_accel[:300])

			#Detecting number of accelerations per shift
			shift_accels = []
			shift_accel_num = []
			shift_starts = []
			shift_starts_globe = []
			shift_ends = []
			shift_end_globe = []
			shift_HR_max = []
			shift_HR_average = []
			shift_HR_std = []


			for shift in shift_indexes:
				#HR Data matching with shifts

				HR_shift_start = np.where(HR_time >= IMU_time[shift[0]])[0][0]
				HR_shift_end = np.where(HR_time >= IMU_time[shift[-1]])[0][0]
				
				HR_shift = HR_data.iloc[HR_shift_start:HR_shift_end,:]
				shift_HR_max.append(max(HR_shift['HR [bpm]']))
				shift_HR_average.append(np.mean(HR_shift['HR [bpm]']))
				shift_HR_std.append(np.std(HR_shift['HR [bpm]']))

				



				accels,_ = find_peaks(trace_accel[shift], height=1, width=100)
				accels = np.array(accels)+shift[0]
				shift_accel_num.append(len(accels))
				shift_accels.append(accels)
				shift_starts.append(IMU_time[shift[0]])
				shift_ends.append(IMU_time[shift[-1]])
				
				shift_starts_globe.append(str(datetime.strptime(str(IMU_start), '%H%M%S')+timedelta.Timedelta(seconds=int(IMU_time[shift[0]]))).split(' ')[-1])
				shift_end_globe.append(str(datetime.strptime(str(IMU_start), '%H%M%S')+timedelta.Timedelta(seconds=int(IMU_time[shift[-1]]))).split(' ')[-1])

				fig.add_shape(
				        type="rect",
				        x0=IMU_time[shift[0]],
				        x1=IMU_time[shift[-1]],
				        y0=0,
				        y1= 35,
				        fillcolor="grey",
				        opacity=0.1)
			
			shift_accels = np.concatenate(shift_accels)
			fig.add_trace(go.Scatter(x=IMU_time[window:], y=trace_accel,
		                mode='lines',
		                marker=dict(
				            color='#ff7f0e',
				            size=3),
		                name=f'Acceleration Impulse {player}', 
		                ))
			fig.add_trace(go.Scatter(x=IMU_time[shift_accels+window], y=trace_accel[shift_accels+window],
		                mode='markers',
		                marker=dict(
				            color='blue',
				            size=3),
		                name=f'Shift Accels {player}', 
		                ))
			fig.add_vline(x=game_start, line_width=3, line_dash="dash", line_color=  '#2ca02c' , annotation_text= 'Game Start', annotation_textangle = 270, annotation_position="top left")
			
			
			fig.add_vline(x=game_end, line_width=3, line_dash="dash", line_color=  '#e377c2' , annotation_text= 'Game End', annotation_textangle = 270, annotation_position="top left")

			fig.add_trace(go.Scatter(x=HR_time, y=HR,
		                mode='markers',
		                marker=dict(
				            color='Red',
				            size=2),
		                name=f'Heart Rate [BPM]{player}'
		                ),secondary_y=True,)
			
			fig.add_trace(go.Scatter(x=stroke_time, y=trace_accel[all_strokes],
		                mode='markers',
		                marker=dict(
				            color='Green',
				            size=3),
		                name='Stroke'
		                ),secondary_y=False,)
			global_time_reindex = global_time.reset_index(drop= True)
			
			
			if markers == True:
				for i in global_time.index.tolist():
					fig.add_vline(x=global_time[i], line_width=1, line_dash="dash", line_color=  '#7f7f7f' , annotation_text= polar_log['Marker Name'][i], annotation_textangle = 90,opacity=0.2)

			fig.update_layout(title = "Para-Hockey In Game Sensor Analysis", 
									xaxis_title = '<b>Time</b> (s)')

			fig.update_yaxes(title_text="<b>Acceleration</b> (m/s^2)", secondary_y=False)
			fig.update_yaxes(range=[60,220],secondary_y=True)
			fig.update_yaxes(range=[-5,10],secondary_y=False)
			fig.update_yaxes(title_text="<b>Beats Per Minute</b> (BPM)", secondary_y=True, showgrid = False)
			fig.update_layout(xaxis=dict(showgrid=False),
		              yaxis=dict(showgrid=False))

			html_file_path = f'{uploaded_data}{game}/exports/{name_last}_{name_first}_gameSummary'
			
			

			if process_all == False: 
				st.plotly_chart(fig)
				
			else: 
				fig.write_html(f"{html_file_path}.html")
				
		
			athlete_export = pd.DataFrame()
			athlete_export['Shift'] = range(1,shift_number+1)
			athlete_export['Event Date'] = date
			athlete_export['first'] = name_first
			athlete_export['last'] = name_last
			athlete_export['Block Start'] = shift_starts
			athlete_export['Block End'] = shift_ends
			athlete_export['Block Duration (s)'] = np.array(shift_ends)-np.array(shift_starts)
			athlete_export['Block Start (global)'] = shift_starts_globe
			athlete_export['Block End (global)'] = shift_end_globe

			athlete_export['Strokes'] = stroke_shift_num
			athlete_export['accelerations'] = shift_accel_num
			athlete_export['Max HR(bpm)'] = shift_HR_max
			athlete_export['Average HR(bpm)'] = shift_HR_average
			athlete_export['HR(bpm) STD'] = shift_HR_std

			athlete_export_app.append(athlete_export)
		except Exception as e:
			st.write(f'An error occured for athlete {name_first} {name_last}: {e}')
			continue
			

	shift_data_export = pd.concat(athlete_export_app, ignore_index=True)
	st.write(shift_data_export)
	
	if process_all == True: 
		shift_data_export.to_csv(f'{uploaded_data}{game}/exports/{game}_summary_data.csv')









