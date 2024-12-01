import csv
import datetime
import time
import multiprocessing
from collections import defaultdict

def parse_float(value):
	try:
		return float(value)
	except:
		return None

def parse_date(value):
	date_formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]
	for fmt in date_formats:
		try:
			return datetime.datetime.strptime(value, fmt)
		except:
			continue
	return None

def read_and_group_data(filename):
	devices_data = defaultdict(list)
	with open(filename, mode='r', newline='', encoding='utf-8') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter='|')
		headers = next(csv_reader)
		for row in csv_reader:
			if len(row) < 10:
				continue
			device = row[1].strip()
			date_str = row[3].strip()

			if not device or not date_str:
				continue
			date = parse_date(date_str)
			if not date:
				continue
			
			ruido = row[7]
			eco2 = row[8]
			etvoc = row[9]
			if not ruido or not eco2 or not etvoc:
				continue


			ruido = parse_float(row[7])
			eco2 = parse_float(row[8]) 
			etvoc = parse_float(row[9])
			if all(value is None for value in [ruido, eco2, etvoc]):
				continue

			devices_data[device].append({
				'date': date,
				'ruido': ruido,
				'eco2': eco2,
				'etvoc': etvoc
			})

	return devices_data

def process_device(device_name, measurements, field):
	intervals = []
	if not measurements:
		return intervals

	last_value = None
	start_time = None
	end_time = None

	for measurement in measurements:
		current_value = measurement.get(field)

		if current_value is None:
			if last_value is not None:
				duration = end_time - start_time
				intervals.append({
					'device': device_name,
					'value': last_value,
					'start_time': start_time,
					'end_time': end_time,
					'duration': duration
				})
				last_value = None
			continue

		if last_value is None or current_value != last_value:
			if last_value is not None:
				duration = end_time - start_time
				intervals.append({
					'device': device_name,
					'value': last_value,
					'start_time': start_time,
					'end_time': end_time,
					'duration': duration
				})

			last_value = current_value
			start_time = measurement['date']
			end_time = measurement['date']
		else:
			end_time = measurement['date']

	if last_value is not None:
		duration = end_time - start_time
		intervals.append({
			'device': device_name,
			'value': last_value,
			'start_time': start_time,
			'end_time': end_time,
			'duration': duration
		})

	return intervals
def worker_process(device_data):
	device_name, measurements = device_data

	sensor_intervals = {
		'ruido': [],
		'eco2': [],
		'etvoc': [],
	}
	sensors = ['ruido', 'eco2', 'etvoc']
	for sensor in sensors:
		sensor_intervals[sensor] = process_device(device_name, measurements, sensor)
	return sensor_intervals

def process_all_devices(devices_data, num_processes):
	device_items = list(devices_data.items())
	with multiprocessing.Pool(processes=num_processes) as pool:
		results = pool.map(worker_process, device_items)
	combined_intervals = {
		'ruido': [],
		'eco2': [],
		'etvoc': []
	}
	for sensor_intervals in results:
		for sensor in combined_intervals:
			combined_intervals[sensor].extend(sensor_intervals[sensor])
	return combined_intervals

def get_top_n_intervals(combined_intervals, n=50):
	top_intervals = {}
	for sensor in combined_intervals:
		intervals = combined_intervals[sensor]
		intervals.sort(key=lambda x: x['duration'], reverse=True)
		top_intervals[sensor] = intervals[:n]
	return top_intervals

def display_intervals(top_intervals):
	for sensor in top_intervals:
		print(f"Top 50 maiores intervalos para {sensor}:")
		for idx, interval in enumerate(top_intervals[sensor], start=1):
			device = interval['device']
			value = interval['value']
			start_time = interval['start_time']
			end_time = interval['end_time']
			duration = interval['duration']

			days = duration.days
			seconds = duration.seconds
			hours = seconds // 3600
			minutes = (seconds % 3600) // 60
			seconds = seconds % 60
			duration = f"{days}d{hours}h{minutes}m{seconds}s"

			print(f"{idx}. Device: {device}, Value: {value}, Start: {start_time}, End: {end_time}, Duration: {duration}")
		print("\n")

def main():
	import argparse
	parser = argparse.ArgumentParser(description='Processamento de dados dos sensores.')
	parser.add_argument('--processes', type=int, default=1, help='Número de processos a serem utilizados.')
	parser.add_argument('--csv_path', type=str, required=True, help='Caminho para o arquivo CSV.')
	args = parser.parse_args()
	num_processes = args.processes
	csv_file_path = args.csv_path
	start_time = time.time()

	print("Lendo e agrupando os dados...")

	devices_data = read_and_group_data(csv_file_path)
	read_time = time.time()

	print(f"Leitura e agrupamento dos dados concluídos em {read_time - start_time:.2f} segundos.")
	print(f"Processando dados com {num_processes} processos...")

	combined_intervals = process_all_devices(devices_data, num_processes)
	process_time = time.time()

	print(f"Processamento dos dados concluído em {process_time - read_time:.2f} segundos.")
	print("Obtendo os 50 maiores intervalos...")

	top_intervals = get_top_n_intervals(combined_intervals, n=50)
	get_top_time = time.time()

	print(f"Obtidos os maiores intervalos em {get_top_time - process_time:.2f} segundos.")
	total_time = get_top_time - start_time

if __name__ == '__main__':
	main()