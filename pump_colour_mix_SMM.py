import time
import math
from rzdt import *
from rzbase import *
import os
import serial.tools.list_ports
from datetime import datetime


class PumpSystem:
	def __init__(self, com_port):
		'''
		Specify the system COM port as str. 
		Opens the serial port and initialises all pumps by cycling through addresses sequentially. 
		Saves a pump list and dict attribute for later use. 
		Designed for a 2 pump system. 
		
		6 Valve Distributor Positions (All Pumps)
		Position 1 = Wash Bottle (H2O)
		Position 2.1 = Red 
		Position 2.2 = Yellow
		Position 3.1 = Blue
		Position 3.2 = Air
		Position 4 = Air
		Position 5 = Chamber 
		Position 6 = Waste Bottle

		'''

		self.pump_list = [1,2]
		self.pump_1 = [1]
		self.pump_2 = [2]

		serialPort = com_port        # Windows only
		serialBaudrate = 9600       # Default baudrate of RzController
		self.logger = RzLogger.get_instance()
		self.default_speed = 440 # in steps/s - comes out to 183 ul/s with 5ml syringe
		self.full_stroke = 11900 # in steps
		self.max_speed = 3000 # in steps/s
		self.pump_dict = {}

		try:
			self.logger.info("---------- Open Serial Port ----------")
			self.conn = RzdtConn(serialPort, serialBaudrate)
			self.logger.info("---------- Initialising all pumps ----------")
			for pump in self.pump_list:
				self.pump_dict[pump] = RzdtDeviceUtil(self.conn, pump)
				self.pump_dict[pump].controller.initialize_system()
				self.pump_dict[pump].wait_and_check_system_ready() #this is needed after every cmd to prevent overflow
				self.pump_dict[pump].controller.set_top_speed_of_plunger(self.max_speed) #set pump to top speed
				self.pump_dict[pump].wait_and_check_system_ready()
				self.pump_dict[pump].controller.query()

		### error handling block included in every function, reports problems to logger and stops machine
		except RzException as e:
			error_code = str(e.error_code)
			self.logger.error(f"RzException: {e} error_code: {error_code}")
			print(error_code)
			self.logger.info("---------- Disconnecting System ----------")
			self.conn.close()

		except Exception as e:
			self.logger.error(f"Exception: {e}")
			print(e)
			self.logger.info("---------- Disconnecting System ----------")
			self.conn.close()

	def close(self):
		self.logger.info("---------- Disconnecting System ----------")
		self.conn.close()

	def move(self, pump_list, pos, speed, valve):
		'''moves the syringe plunger for pumps specified in pump list (list of addresses as ints) to the specified position in
		steps (int) at the specified speed in steps per second (int) at the specified valve address (int)'''
		try:
			for pump in pump_list:
				self.pump_dict[pump].wait_and_check_system_ready()
			for pump in pump_list:
				self.pump_dict[pump].controller.set_top_speed_of_plunger(speed) #set pump to speed
			for pump in pump_list:
				self.pump_dict[pump].wait_and_check_system_ready()
			for pump in pump_list:
				self.pump_dict[pump].controller.move_valve(valve, RzDef.AUTO) #move valve to pos
			for pump in pump_list:
				self.pump_dict[pump].wait_and_check_system_ready()
			for pump in pump_list:
				self.pump_dict[pump].controller.move_plunger(pos) #move plunger
			time.sleep(self.full_stroke/speed-1) #needed to avoid timeout, assuming full strokes
			for pump in pump_list:
				self.pump_dict[pump].wait_and_check_system_ready()

		except RzException as e:
			error_code = str(e.error_code)
			self.logger.error(f"RzException: {e} error_code: {error_code}")
			print([e, error_code])
			self.close()

		except Exception as e:
			self.logger.error(f"Exception: {e}")
			print(e)
			self.close()

	def empty_lines(self):
		for i in [2,3]:
			self.move(self.pump_list, self.full_stroke, self.max_speed, i)
			self.move(self.pump_list, 0, self.max_speed, 6)
		print("Lines Emptied")

	def flush_syringes(self, pump_list):
		'''flushes specified syringes (specify as list containing desired numerical pump addresses) with a full stroke of wash solution
		and dispenses to waste'''
		
		print("Flushing syringes")
		self.logger.info("---------- Flushing pumps "+str(pump_list)+"----------")
		self.move(pump_list, self.full_stroke, self.max_speed, 1) #draw wash solution
		self.move(pump_list, 0, self.max_speed, 6) #fully dispense to waste
		print("Flush finished")


	def clean_system(self):
		'''throws out the chamber and flushes with water'''
		
		print("Cleaning system")

		### empty the chamber
		for i in range(3):
			self.move(self.pump_list, self.full_stroke, self.max_speed, 5) #draw colour solution
			self.move(self.pump_list, 0, int(self.default_speed/2), 6) #dispose to waste bottle

		### rinse the syringes
		self.flush_syringes(self.pump_pairs[0])
		
		### fill the chamber with water and dump to waste
		for i in range(2):
			self.move(self.pump_list, self.full_stroke, self.max_speed, 1) #draw water
			self.move(self.pump_list, 0, self.max_speed, 5) #dump into chamber

		for i in range(3):
			self.move(self.pump_list, self.full_stroke, self.max_speed, 5) #draw water from chamber
			self.move(self.pump_list, 0, self.max_speed, 6) #dump into waste

		print("Cleaning finished")
		

	def colour_mix(self, red=0, yellow=5, blue=5):
		'''Sepcify the colour mix as a combination of red (mL), yellow (mL) and blue (mL).
		Constraint: The maximum value of any colour is 5mL - cannot do more than that.'''

		### fill red
		self.move(self.pump_1, red, self.max_speed, 2) #draw from sample bottle red
		self.move(self.pump_1, 0, self.max_speed, 5) #put into chamber

		### fill yellow
		self.move(self.pump_2, yellow, self.max_speed, 2) #draw from sample bottle yellow
		self.move(self.pump_2, 0, self.max_speed, 5) #draw from sample bottle red

		### fill blue
		self.move(self.pump_1, blue, self.max_speed, 3) #draw from sample bottle blue
		self.move(self.pump_1, 0, self.max_speed, 5) #draw from sample bottle red



if __name__ == "__main__":
	
	### List and select the COM ports available and automatically use the port number
	ports = list(serial.tools.list_ports.comports())
	for p in ports:
		com_num = "COM5" ###p[0]
		#print(com_num)
		
	system = PumpSystem(com_num)

	### Flush Syringes
	#system.flush_syringes(system.pump_list)

	### Cleaning 
	# system.clean_system()
	x = 2
	y = 8
	z = 0

	system.colour_mix(red = 4, yellow = 1, blue = 3) 

	### drain the lines
	# system.empty_lines()

	system.close()