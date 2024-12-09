# use pyvisa or similar package
import pyvisa
import time
import os
import math
import pandas
import matplotlib.pyplot as plt


class MagneticMeasurer:
    def __init__(self, SCPI_server_IP='192.168.96.135'):

        SCPI_Port = '5025'

        self.SCPI_timeout = 20000 # milliseconds
        self.VISA_server = f'TCPIP::{SCPI_server_IP}::{SCPI_Port}::SOCKET'

        self.sweep_type = 'LOG'
        self.receiver_bandwidth = '300Hz'
        self.source_power = '13'

        self.visa_session = self.new_visa_session()

    def new_visa_session(self):
        print(f'Trying to connect to VISA resource: {self.VISA_server}. Be sure that IP address and port number are correct!' )
        visa_session = pyvisa.ResourceManager().open_resource(self.VISA_server)
        visa_session.timeout = self.SCPI_timeout
        visa_session.read_termination = '\n'

        print(f'SCPI client connected to SCPI server: {visa_session.query("*IDN?")}')

        return visa_session

    def is_calibrated(self):
        isCalibrated = visa_session.query(':SENS:CORR:FULL:ACT?')
        return isCalibrated == "1"

    def calibrate(self, calibration_file, force_calibration=False, calibration_load=100):

        isCalibrated = self.visa_session.query(':SENS:CORR:FULL:ACT?') == "1"
        print(f"start isCalibrated raw: {self.visa_session.query(':SENS:CORR:FULL:ACT?')}")
        print(f"start isCalibrated: {isCalibrated}")
        if not isCalibrated or force_calibration:
            if os.path.exists(calibration_file) and not force_calibration:
                print("Loading calibration")

                calibration_loaded = False
                while not calibration_loaded:
                    self.visa_session.write(f':MMEM:LOAD:CORR:FULL "{calibration_file}"\n')
                    print(self.visa_session.write('*WAI\n'))
                    print(self.visa_session.query('*OPC?'))

                    # Testing
                    try:
                        print("Testing")
                        self.take_Rs_Ls_measurement(
                            start_frequency=10000,
                            stop_frequency=11000,
                            number_of_measurement_cycles=1
                        )
                        calibration_loaded = True
                    except pyvisa.errors.VisaIOError:
                        print("Timeout, retrying")
                        calibration_loaded = False
                    
            else:
                print("Calibrating")
                self.visa_session.write(":SOUR:POW 13\n");
                self.visa_session.write(f":SENS:CORR:LOAD {calibration_load}\n");
                self.visa_session.write(":CALC:ZPAR:DEF Z\n");
                method = self.visa_session.query(':SENS:Z:METH?')
                par = self.visa_session.query(':CALC:PAR:DEF?')
                print(method)
                print(par)
                amplitude = self.visa_session.query(':SOUR:POW?')
                print(amplitude)
                zparameter = self.visa_session.query(':CALC:ZPAR:DEF?')
                print(zparameter)
                print(self.visa_session.query(':SENS:CORR:LOAD?'))
                # assert 0
                print(f"isCalibrated: {isCalibrated}")
                input("Leave adapter open and press Enter to continue...")
                print(self.visa_session.write(':SENS:CORR:FULL:OPEN\n'))
                print(self.visa_session.write('*WAI\n'))
                self.visa_session.query('*OPC?')
                input("Enter short circuit board and press Enter to continue...")
                print(self.visa_session.write(':SENS:CORR:FULL:SHOR\n'))
                print(self.visa_session.write('*WAI\n'))
                self.visa_session.query('*OPC?')
                input(f"Enter {calibration_load} Ohm board and press Enter to continue...")
                print(self.visa_session.write(':SENS:CORR:FULL:LOAD\n'))
                print(self.visa_session.write('*WAI\n'))
                self.visa_session.query('*OPC?')
                isCalibrated = self.visa_session.query(':SENS:CORR:FULL:ACT?') == "1"
                print(f"isCalibrated: {isCalibrated}")
                if isCalibrated:
                    print(self.visa_session.query(':SENS:CORR:FULL:DATA:OPEN?'))
                    print(self.visa_session.query(':SENS:CORR:FULL:DATA:SHOR?'))
                    print(self.visa_session.query(':SENS:CORR:FULL:DATA:LOAD?'))

                    # Testing
                    try:
                        print("Testing")
                        data = self.take_Rs_Ls_measurement(
                            start_frequency=100,
                            stop_frequency=1000,
                            number_of_measurement_cycles=1
                        )
                        print(data["resistance"].iloc[0])
                        error = abs(data["resistance"].iloc[0] - calibration_load) / calibration_load
                        print(error)
                        if (error < 0.0001):
                            calibration_loaded = True
                        else:
                            assert calibration_loaded, "Load measured is different from calibration_load"
                    except pyvisa.errors.VisaIOError:
                        print("Timeout, retrying")
                        calibration_loaded = False

                    print(f"Storing calibration in {calibration_file}")
                    print(self.visa_session.write(f':MMEM:STOR:CORR "{calibration_file}"\n'))

                else:
                    assert 0
        isCalibrated = self.visa_session.query(':SENS:CORR:FULL:ACT?') == "1"
        print(f"isCalibrated raw: {self.visa_session.query(':SENS:CORR:FULL:ACT?')}")
        print(f"isCalibrated: {isCalibrated}")
        return isCalibrated

    def take_Rs_Ls_measurement(self, start_frequency=10000, stop_frequency=1000000, number_of_measurement_cycles=2, number_of_measurement_points=201):

        self.visa_session.write(f':SENS:FREQ:STAR {start_frequency}')
        self.visa_session.write(f':SENS:FREQ:STOP {stop_frequency}')
        self.visa_session.write(':CALC:ZPAR:DEF Z')
        self.visa_session.write(':CALC:PAR:DEF Z') # configuring 'one-port' impedance measurement
        self.visa_session.write(':SENS:Z:METH IAD\n')
        self.visa_session.write(f':SENS:SWE:POIN {number_of_measurement_points}')
        self.visa_session.write(f':SENS:SWE:TYPE {self.sweep_type}')
        self.visa_session.write(f':SENS:BAND {self.receiver_bandwidth}')

        # self.visa_session.write(":CALC:FORM SLIN") # linear magnitude in Ohms + phase(deg)
        self.visa_session.write(":CALC:FORM SCOM")

        self.visa_session.write(':TRIG:SOUR BUS')  # Intializes trigger system to use BUS - to be used in combination with TRIG:SING and OPC
        self.visa_session.write(':INIT:CONT ON') # Sets the trigger in continous mode. This way after a measurement the trigges gets back in state "ready" and waits for a further measurement.

        measurements = pandas.DataFrame()
        for measurement_index in range(number_of_measurement_cycles):
            self.visa_session.write(':TRIG:SING')
            self.visa_session.query('*OPC?')

            allResults = self.visa_session.query(':CALC:DATA:SDAT?')
            frequencyValues = self.visa_session.query(':SENS:FREQ:DATA?')

            allResults_list_raw = list(map(float, allResults.split(",")))
            real_data = allResults_list_raw[:number_of_measurement_points]
            imaginary_data = allResults_list_raw[number_of_measurement_points:]

            frequencies = list(map(float, frequencyValues.split(",")))

            for index in range(number_of_measurement_points):
                angular_frequency = 2 * math.pi * frequencies[index]
                new_measurement = pandas.DataFrame([{
                    "measurement_index": measurement_index,
                    "frequency": frequencies[index],
                    "resistance": real_data[index],
                    "inductance": imaginary_data[index] / angular_frequency,
                }])
                measurements = pandas.concat([measurements, new_measurement], ignore_index=True)
        return measurements

    def take_Z_phase_measurement(self, start_frequency=100, stop_frequency=40000000, number_of_measurement_cycles=1, number_of_measurement_points=201):
        self.visa_session.write(f':SENS:FREQ:STAR {start_frequency}')
        self.visa_session.write(f':SENS:FREQ:STOP {stop_frequency}')
        self.visa_session.write(':CALC:ZPAR:DEF Z')
        self.visa_session.write(':CALC:PAR:DEF Z') # configuring 'one-port' impedance measurement
        self.visa_session.write(':SENS:Z:METH IAD\n')
        self.visa_session.write(f':SENS:SWE:POIN {number_of_measurement_points}')
        self.visa_session.write(f':SENS:SWE:TYPE {self.sweep_type}')
        self.visa_session.write(f':SENS:BAND {self.receiver_bandwidth}')

        self.visa_session.write(":CALC:FORM SLIN") # linear magnitude in Ohms + phase(deg)
        # self.visa_session.write(":CALC:FORM SCOM")

        self.visa_session.write(':TRIG:SOUR BUS')  # Intializes trigger system to use BUS - to be used in combination with TRIG:SING and OPC
        self.visa_session.write(':INIT:CONT ON') # Sets the trigger in continous mode. This way after a measurement the trigges gets back in state "ready" and waits for a further measurement.

        measurements = pandas.DataFrame()
        for measurement_index in range(number_of_measurement_cycles):
            self.visa_session.write(':TRIG:SING')
            self.visa_session.query('*OPC?')

            allResults = self.visa_session.query(':CALC:DATA:SDAT?')
            frequencyValues = self.visa_session.query(':SENS:FREQ:DATA?')

            allResults_list_raw = list(map(float, allResults.split(",")))
            magnitude_data = allResults_list_raw[:number_of_measurement_points]
            phase_data = allResults_list_raw[number_of_measurement_points:]

            frequencies = list(map(float, frequencyValues.split(",")))

            for index in range(number_of_measurement_points):
                angular_frequency = 2 * math.pi * frequencies[index]
                new_measurement = pandas.DataFrame([{
                    "measurement_index": measurement_index,
                    "frequency": frequencies[index],
                    "magnitude": magnitude_data[index],
                    "phase": phase_data[index],
                }])
                measurements = pandas.concat([measurements, new_measurement], ignore_index=True)
        return measurements

    def take_Cp_measurement(self, start_frequency=10000, stop_frequency=1000000, number_of_measurement_cycles=2, number_of_measurement_points=201):

        self.visa_session.write(f':SENS:FREQ:STAR {start_frequency}')
        self.visa_session.write(f':SENS:FREQ:STOP {stop_frequency}')
        self.visa_session.write(':CALC:PAR:DEF Z')
        self.visa_session.write(':CALC:ZPAR:DEF Cs')
        self.visa_session.write(':SENS:Z:METH IAD\n')
        self.visa_session.write(f':SENS:SWE:POIN {number_of_measurement_points}')
        self.visa_session.write(f':SENS:SWE:TYPE {self.sweep_type}')
        self.visa_session.write(f':SENS:BAND {self.receiver_bandwidth}')

        self.visa_session.write(':CALC:FORM REAL')

        self.visa_session.write(':TRIG:SOUR BUS')  # Intializes trigger system to use BUS - to be used in combination with TRIG:SING and OPC
        self.visa_session.write(':INIT:CONT ON') # Sets the trigger in continous mode. This way after a measurement the trigges gets back in state "ready" and waits for a further measurement.

        measurements = pandas.DataFrame()
        for measurement_index in range(number_of_measurement_cycles):
            self.visa_session.write(':TRIG:SING')
            self.visa_session.query('*OPC?')

            allResults = self.visa_session.query(':CALC:DATA:SDAT?')
            frequencyValues = self.visa_session.query(':SENS:FREQ:DATA?')

            allResults_list_raw = list(map(float, allResults.split(",")))
            parallel_capacitance_data = allResults_list_raw[:number_of_measurement_points]

            frequencies = list(map(float, frequencyValues.split(",")))

            for index in range(number_of_measurement_points):
                angular_frequency = 2 * math.pi * frequencies[index]
                new_measurement = pandas.DataFrame([{
                    "measurement_index": measurement_index,
                    "frequency": frequencies[index],
                    "capacitance": parallel_capacitance_data[index]
                }])
                measurements = pandas.concat([measurements, new_measurement], ignore_index=True)
        return measurements

    def close_visa_session(self):
        self.visa_session.close()

    def plot_RL(self, data, plot_resistance=True, resistance_label="Resistance", plot_inductance=True, inductance_label="Inductance"):
        fig, ax1 = plt.subplots()

        grouped = data[['resistance', 'inductance', 'frequency']].groupby(['frequency'], as_index=False)
        print(grouped)
        averaged_measurement = grouped.mean()
        print(averaged_measurement)

        if plot_resistance and not plot_inductance:
            first_data = 'resistance'
            first_label = resistance_label
            second_data = None
            second_label = None
        elif not plot_resistance and plot_inductance:
            first_data = 'inductance'
            first_label = inductance_label
            second_data = None
            second_label = None
        elif plot_resistance and plot_inductance:
            first_data = 'resistance'
            first_label = resistance_label
            second_data = 'inductance'
            second_label = inductance_label
        
        color = 'tab:red'
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel(first_label, color=color)
        ax1.plot(averaged_measurement['frequency'], averaged_measurement[first_data], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xscale('log')

        if second_data is not None:
            ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel(second_label, color=color)  # we already handled the x-label with ax1
            ax2.plot(averaged_measurement['frequency'], averaged_measurement[second_data], color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_xscale('log')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    def plot_Z(self, data):
        fig, ax1 = plt.subplots()

        grouped = data[['magnitude', 'phase', 'frequency']].groupby(['frequency'], as_index=False)
        print(grouped)
        averaged_measurement = grouped.mean()
        print(averaged_measurement)

        first_data = 'magnitude'
        first_label = "Impedance magnitude"
        second_data = 'phase'
        second_label = "Phase"
    
        color = 'tab:red'
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel(first_label, color=color)
        ax1.plot(averaged_measurement['frequency'], averaged_measurement[first_data], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xscale('log')

        if second_data is not None:
            ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel(second_label, color=color)  # we already handled the x-label with ax1
            ax2.plot(averaged_measurement['frequency'], averaged_measurement[second_data], color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_xscale('log')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    def plot(self, data, column, label):
        fig, ax1 = plt.subplots()

        grouped = data[[column, 'frequency']].groupby(['frequency'], as_index=False)
        print(grouped)
        averaged_measurement = grouped.mean()
        print(averaged_measurement)


        color = 'tab:red'
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel(label, color=color)
        ax1.plot(averaged_measurement['frequency'], averaged_measurement[column], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xscale('log')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

# if __name__ == "__main__":
    # magnetic_measurer = MagneticMeasurer()
    # data = magnetic_measurer.take_Cp_measurement()
    # magnetic_measurer.plot(data, "capacitance", "Parallel capacitance")
    # magnetic_measurer.calibrate("C:\\Users\\Alfonso\\Lab\\calibrations\\isi_board.mcalx")
    # data = magnetic_measurer.take_Z_phase_measurement(stop_frequency=40000000)
    # magnetic_measurer.plot_Z(data)
    # magnetic_measurer.calibrate("C:\\Users\\Alfonso\\Lab\\calibrations\\isi_board.mcalx", force_calibration=True, calibration_load=100.3)
