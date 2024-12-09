import Bode100Analyzer
import pyvisa
import ast
import pathlib
import os
import pandas
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import fsolve, least_squares
import numpy


class MagneticCharacterizer:
    def  __init__(self, reference="temp"):
        self.reference = reference
        self.bode_100 = Bode100Analyzer.MagneticMeasurer()
        self.bode_100.calibrate(f"{pathlib.Path(__file__).parent.resolve()}\\calibrations\\isi_board.mcalx")

        self.output_path = pathlib.Path(__file__).parent.resolve() / "output"
        pathlib.Path(self.output_path).mkdir(parents=True, exist_ok=True)

    def extract_value_at_frequency(self, data, frequency, parameter="inductance"):
        closest_frequency = None
        closest_value = None
        minimum_error = 100
        for _, row in data.iterrows():
            error = abs(row["frequency"] - frequency) / frequency
            if error < minimum_error:
                closest_frequency = row["frequency"]
                closest_value = row[parameter]
                minimum_error = error
                if error == 0:
                    break

        return (closest_frequency, closest_value)

    def average_measurements(self, data):
        grouped = data.groupby(['frequency'], as_index=False)
        print(grouped)
        averaged_measurements = grouped.mean()
        print(averaged_measurements)
        return averaged_measurements

    def detect_zero_crossing(self, data):
        data = self.average_measurements(data)
        resonances = []
        phase_slopes = []


        peak_indexes, properties = find_peaks(data["magnitude"], prominence=2)
        for index in peak_indexes:
            resonances.append({"frequency": data.loc[index, "frequency"], "impedance_magnitude": data.loc[index, "magnitude"], "type": "local maximum"})

        peak_indexes, properties = find_peaks(-data["magnitude"], prominence=2)
        for index in peak_indexes:
            resonances.append({"frequency": data.loc[index, "frequency"], "impedance_magnitude": data.loc[index, "magnitude"], "type": "local minimum"})

        resonances = sorted(resonances, key=lambda resonance: resonance["frequency"])

        return resonances


    def characterize_inductance(self):
        input("Place secondary in open circuit, setup primary up for measurement and press Enter to continue...")
        data_Lp_OS = self.bode_100.take_Rs_Ls_measurement(
            start_frequency=10000,
            stop_frequency=1000000,
            number_of_measurement_cycles=2
        )

        input("Place primary in open circuit, setup secondary up for measurement and press Enter to continue...")
        data_Ls_OP = self.bode_100.take_Rs_Ls_measurement(
            start_frequency=10000,
            stop_frequency=1000000,
            number_of_measurement_cycles=2
        )

        input("Place circuit in cummulative flux mode and press Enter to continue...")
        data_Lcum = self.bode_100.take_Rs_Ls_measurement(
            start_frequency=10000,
            stop_frequency=1000000,
            number_of_measurement_cycles=2
        )

        input("Place circuit in differential flux mode and press Enter to continue...")
        data_Ldif = self.bode_100.take_Rs_Ls_measurement(
            start_frequency=10000,
            stop_frequency=1000000,
            number_of_measurement_cycles=2
        )

    def characterize_inductance_basic(self, allow_use_cache=False):
        magnetizing_inductance_filepath  = self.output_path / f"{self.reference}_basic_inductance_characterization_magnetizing_inductance.csv"
        leakage_inductance_filepath  = self.output_path / f"{self.reference}_basic_inductance_characterization_leakage_inductance.csv"


        if allow_use_cache and os.path.exists(magnetizing_inductance_filepath):
            magnetizing_inductance = pandas.read_csv(magnetizing_inductance_filepath)
        else:
            input("Place secondary in open circuit, setup primary up for measurement and press Enter to continue...")
            data_Lp_OS = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=1000000,
                number_of_measurement_cycles=2
            )
            magnetizing_inductance = data_Lp_OS
            magnetizing_inductance.to_csv(magnetizing_inductance_filepath, index=False)

        if allow_use_cache and os.path.exists(leakage_inductance_filepath):
            leakage_inductance = pandas.read_csv(leakage_inductance_filepath)
        else:
            input("Place secondary in short circuit, setup primary up for measurement and press Enter to continue...")
            data_Lp_SS = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=1000000,
                number_of_measurement_cycles=2
            )
            leakage_inductance = data_Lp_SS
            leakage_inductance.to_csv(leakage_inductance_filepath, index=False)

        measured_frequency, magnetizing_inductance_at_10kHz = self.extract_value_at_frequency(magnetizing_inductance, 10000)
        print(f"magnetizing_inductance_at_10kHz : {magnetizing_inductance_at_10kHz}")
        measured_frequency, leakage_inductance_10kHz = self.extract_value_at_frequency(leakage_inductance, 10000)
        print(f"leakage_inductance_10kHz : {leakage_inductance_10kHz}")

        self.bode_100.plot_RL(
            data=magnetizing_inductance,
            plot_resistance=False,
            resistance_label="Resistance",
            plot_inductance=True,
            inductance_label="Magnetizing Inductance"
        )

        self.bode_100.plot_RL(
            data=leakage_inductance,
            plot_resistance=False,
            resistance_label="Resistance",
            plot_inductance=True,
            inductance_label="Leakage Inductance"
        )

    def characterize_inductance_medium(self, allow_use_cache=False):
        data_Lp_OS_filepath  = self.output_path / f"{self.reference}_medium_inductance_characterization_data_Lp_OS.csv"
        data_Lp_SS_filepath  = self.output_path / f"{self.reference}_medium_inductance_characterization_data_Lp_SS.csv"

        if allow_use_cache and os.path.exists(data_Lp_OS_filepath):
            data_Lp_OS = pandas.read_csv(data_Lp_OS_filepath)
        else:
            input("Place secondary in open circuit, setup primary up for measurement and press Enter to continue...")
            data_Lp_OS = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=1000000,
                number_of_measurement_cycles=2
            )
            measured_frequency, data_Lp_OS_at_10kHz = self.extract_value_at_frequency(data_Lp_OS, 10000)
            print(data_Lp_OS_at_10kHz)
            data_Lp_OS.to_csv(data_Lp_OS_filepath, index=False)

        if allow_use_cache and os.path.exists(data_Lp_SS_filepath):
            data_Lp_SS = pandas.read_csv(data_Lp_SS_filepath)
        else:
            input("Place secondary in short circuit, setup primary up for measurement and press Enter to continue...")
            data_Lp_SS = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=1000000,
                number_of_measurement_cycles=2
            )
            measured_frequency, data_Lp_SS_at_10kHz = self.extract_value_at_frequency(data_Lp_SS, 10000)
            print(data_Lp_SS_at_10kHz)
            data_Lp_SS.to_csv(data_Lp_SS_filepath, index=False)

        temp_data = data_Lp_OS[["frequency"]].copy()
        temp_data["Lp_OS"] = data_Lp_OS["inductance"]
        temp_data["Lp_SS"] = data_Lp_SS["inductance"]

        temp_data["coupling_coefficient"] = temp_data.apply(lambda row: math.sqrt(1 - row["Lp_SS"] / row["Lp_OS"]), axis=1)
        temp_data["leakage_inductance"] = temp_data.apply(lambda row: row["Lp_SS"] / row["coupling_coefficient"], axis=1)
        temp_data["magnetizing_inductance"] = temp_data.apply(lambda row: row["Lp_OS"] * (1 + row["coupling_coefficient"]) / 2, axis=1)
        measured_frequency, leakage_inductance_at_10kHz = self.extract_value_at_frequency(temp_data, 10000, parameter="leakage_inductance")
        measured_frequency, magnetizing_inductance_at_10kHz = self.extract_value_at_frequency(temp_data, 10000, parameter="magnetizing_inductance")
        print(leakage_inductance_at_10kHz)
        print(magnetizing_inductance_at_10kHz)

        self.bode_100.plot(
            data=temp_data,
            column="magnetizing_inductance",
            label="Magnetizing Inductance",
        )
        self.bode_100.plot(
            data=temp_data,
            column="leakage_inductance",
            label="Leakage Inductance",
        )

    def characterize_inductance_advanced(self, allow_use_cache=False):
        data_Lp_OS_filepath  = self.output_path / f"{self.reference}_advanced_inductance_characterization_data_Lp_OS.csv"
        data_Ls_OP_filepath  = self.output_path / f"{self.reference}_advanced_inductance_characterization_data_Ls_OP.csv"
        data_Lp_SS_filepath  = self.output_path / f"{self.reference}_advanced_inductance_characterization_data_Lp_SS.csv"
        data_Lcum_filepath  = self.output_path / f"{self.reference}_advanced_inductance_characterization_data_Lcum.csv"
        data_Ldif_filepath  = self.output_path / f"{self.reference}_advanced_inductance_characterization_data_Ldif.csv"

        magnetizing_inductance_filepath  = self.output_path / f"{self.reference}_advanced_inductance_characterization_magnetizing_inductance.csv"
        leakage_inductance_filepath  = self.output_path / f"{self.reference}_advanced_inductance_characterization_leakage_inductance.csv"
        coupling_coefficient_filepath  = self.output_path / f"{self.reference}_advanced_inductance_characterization_coupling_coefficient.csv"

        if allow_use_cache and os.path.exists(data_Lp_OS_filepath):
            data_Lp_OS = pandas.read_csv(data_Lp_OS_filepath)
        else:
            input("Place secondary in open circuit, setup primary up for measurement and press Enter to continue...")
            data_Lp_OS = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=1000000,
                number_of_measurement_cycles=2
            )
            measured_frequency, data_Lp_OS_at_10kHz = self.extract_value_at_frequency(data_Lp_OS, 10000)
            print(data_Lp_OS_at_10kHz)
            data_Lp_OS.to_csv(data_Lp_OS_filepath, index=False)

        if allow_use_cache and os.path.exists(data_Lp_SS_filepath):
            data_Lp_SS = pandas.read_csv(data_Lp_SS_filepath)
        else:
            input("Place secondary in short circuit, setup primary up for measurement and press Enter to continue...")
            data_Lp_SS = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=1000000,
                number_of_measurement_cycles=2
            )
            measured_frequency, data_Lp_SS_at_10kHz = self.extract_value_at_frequency(data_Lp_SS, 10000)
            print(data_Lp_SS_at_10kHz)
            data_Lp_SS.to_csv(data_Lp_SS_filepath, index=False)

        if allow_use_cache and os.path.exists(data_Ls_OP_filepath):
            data_Ls_OP = pandas.read_csv(data_Ls_OP_filepath)
        else:
            input("Place primary in open circuit, setup secondary up for measurement and press Enter to continue...")
            data_Ls_OP = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=1000000,
                number_of_measurement_cycles=2
            )
            measured_frequency, data_Ls_OP_at_10kHz = self.extract_value_at_frequency(data_Ls_OP, 10000)
            print(data_Ls_OP_at_10kHz)
            data_Ls_OP.to_csv(data_Ls_OP_filepath, index=False)


        if allow_use_cache and os.path.exists(data_Lcum_filepath):
            data_Lcum = pandas.read_csv(data_Lcum_filepath)
        else:
            input("Place circuit in cummulative flux mode and press Enter to continue...")
            data_Lcum = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=1000000,
                number_of_measurement_cycles=2
            )
            measured_frequency, data_Lcum_at_10kHz = self.extract_value_at_frequency(data_Lcum, 10000)
            print(data_Lcum_at_10kHz)
            data_Lcum.to_csv(data_Lcum_filepath, index=False)


        if allow_use_cache and os.path.exists(data_Ldif_filepath):
            data_Ldif = pandas.read_csv(data_Ldif_filepath)
        else:
            input("Place circuit in differential flux mode and press Enter to continue...")
            data_Ldif = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=1000000,
                number_of_measurement_cycles=2
            )
            measured_frequency, data_Ldif_at_10kHz = self.extract_value_at_frequency(data_Ldif, 10000)
            print(data_Ldif_at_10kHz)
            data_Ldif.to_csv(data_Ldif_filepath, index=False)


        temp_data = data_Lp_OS[["frequency"]].copy()
        temp_data["Lp_OS"] = data_Lp_OS["inductance"]
        temp_data["Lp_SS"] = data_Lp_SS["inductance"]
        temp_data["Ls_OP"] = data_Ls_OP["inductance"]
        temp_data["Lcum"] = data_Lcum["inductance"]
        temp_data["Ldif"] = data_Ldif["inductance"]
        print(temp_data[["Lp_SS", "Lp_OS"]])
        temp_data["coupling_coefficient"] = temp_data.apply(lambda row: math.sqrt(1 - row["Lp_SS"] / row["Lp_OS"]), axis=1)
        temp_data["n"] = temp_data.apply(lambda row: math.sqrt(row["Lp_OS"] / row["Ls_OP"]), axis=1)

        temp_data["Lmp"] = temp_data.apply(lambda row: (row["Lcum"] - row["Ldif"]) / (4 * row["n"]), axis=1)
        temp_data["Lwp"] = temp_data.apply(lambda row: row["Lp_OS"] - row["Lmp"], axis=1)
        temp_data["Lws"] = temp_data.apply(lambda row: row["Ls_OP"] - row["n"] * row["Lmp"], axis=1)


    def characterize_capacitance_medium(self, allow_use_cache=False):
        #  According to Section III of https://sci-hub.st/https://ieeexplore.ieee.org/abstract/document/746603

        data_Lsc_filepath  = self.output_path / f"{self.reference}_medium_capacitance_characterization_data_Lsc.csv"
        data_L01_filepath  = self.output_path / f"{self.reference}_medium_capacitance_characterization_data_L01.csv"
        data_L02_filepath  = self.output_path / f"{self.reference}_medium_capacitance_characterization_data_L02.csv"
        data_cm1_filepath  = self.output_path / f"{self.reference}_medium_capacitance_characterization_data_cm1.csv"
        data_cm2_filepath  = self.output_path / f"{self.reference}_medium_capacitance_characterization_data_cm2.csv"
        data_cm3_filepath  = self.output_path / f"{self.reference}_medium_capacitance_characterization_data_cm3.csv"
        data_cm4_filepath  = self.output_path / f"{self.reference}_medium_capacitance_characterization_data_cm4.csv"
        data_cm5_filepath  = self.output_path / f"{self.reference}_medium_capacitance_characterization_data_cm5.csv"
        data_cm6_filepath  = self.output_path / f"{self.reference}_medium_capacitance_characterization_data_cm6.csv"


        if allow_use_cache and os.path.exists(data_Lsc_filepath):
            data_Lsc = pandas.read_csv(data_Lsc_filepath)
        else:
            input("Place secondary in short circuit, setup primary up for measurement and press Enter to continue...")
            data_Lsc = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=40000000,
                number_of_measurement_cycles=2
            )
            data_Lsc.to_csv(data_Lsc_filepath, index=False)
        # self.bode_100.plot(data_Lsc, "inductance", "Parallel capacitance")

        if allow_use_cache and os.path.exists(data_L01_filepath):
            data_L01 = pandas.read_csv(data_L01_filepath)
        else:
            input("Place secondary in open circuit, setup primary up for measurement and press Enter to continue...")
            data_L01 = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=40000000,
                number_of_measurement_cycles=2
            )
            data_L01.to_csv(data_L01_filepath, index=False)

        if allow_use_cache and os.path.exists(data_L02_filepath):
            data_L02 = pandas.read_csv(data_L02_filepath)
        else:
            input("Place primary in open circuit, setup secondary up for measurement and press Enter to continue...")
            data_L02 = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=40000000,
                number_of_measurement_cycles=2
            )
            data_L02.to_csv(data_L02_filepath, index=False)

        if allow_use_cache and os.path.exists(data_cm1_filepath):
            data_cm1 = pandas.read_csv(data_cm1_filepath)
        else:
            input("Place secondary and primary in short circuit, connect each to one port and press Enter to continue...")
            data_cm1 = self.bode_100.take_Cp_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_cm1.to_csv(data_cm1_filepath, index=False)
        # self.bode_100.plot(data_cm1, "capacitance", "Parallel capacitance")

        if allow_use_cache and os.path.exists(data_cm2_filepath):
            data_cm2 = pandas.read_csv(data_cm2_filepath)
        else:
            input("Connect output of primary with output of secondary, setup primary up for measurement and press Enter to continue...")
            data_cm2 = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_cm2.to_csv(data_cm2_filepath, index=False)

        if allow_use_cache and os.path.exists(data_cm3_filepath):
            data_cm3 = pandas.read_csv(data_cm3_filepath)
        else:
            input("Connect output of primary with input of secondary, setup primary up for measurement and press Enter to continue...")
            data_cm3 = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_cm3.to_csv(data_cm3_filepath, index=False)

        if allow_use_cache and os.path.exists(data_cm4_filepath):
            data_cm4 = pandas.read_csv(data_cm4_filepath)
        else:
            input("Connect input of primary with input of secondary, short circuit secondary, setup primary up for measurement and press Enter to continue...")
            data_cm4 = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_cm4.to_csv(data_cm4_filepath, index=False)

        if allow_use_cache and os.path.exists(data_cm5_filepath):
            data_cm5 = pandas.read_csv(data_cm5_filepath)
        else:
            input("Connect output of secondary with output of primary, short circuit primary, setup secondary up for measurement and press Enter to continue...")
            data_cm5 = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_cm5.to_csv(data_cm5_filepath, index=False)

        if allow_use_cache and os.path.exists(data_cm6_filepath):
            data_cm6 = pandas.read_csv(data_cm6_filepath)
        else:
            input("Connect input of secondary with input of primary, short circuit primary, setup secondary up for measurement and press Enter to continue...")
            data_cm6 = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_cm6.to_csv(data_cm6_filepath, index=False)

        _, Lsc_at_10kHz = self.extract_value_at_frequency(data_Lsc, 10000)
        _, L01_at_10kHz = self.extract_value_at_frequency(data_L01, 10000)
        _, L02_at_10kHz = self.extract_value_at_frequency(data_L02, 10000)
        _, cm1_at_10kHz = self.extract_value_at_frequency(data_cm1, 10000, parameter="capacitance")
        # self.bode_100.plot_Z(data=data_cm3)
        cm2_resonances = self.detect_zero_crossing(data=data_cm2)
        assert len(cm2_resonances) >= 1
        cm3_resonances = self.detect_zero_crossing(data=data_cm3)
        assert len(cm3_resonances) >= 1
        cm4_resonances = self.detect_zero_crossing(data=data_cm4)
        assert len(cm4_resonances) >= 1
        cm5_resonances = self.detect_zero_crossing(data=data_cm5)
        print(cm5_resonances)
        assert len(cm5_resonances) >= 1
        cm6_resonances = self.detect_zero_crossing(data=data_cm6)
        print(cm6_resonances)
        assert len(cm6_resonances) >= 1



        turns_ratio = math.sqrt(L02_at_10kHz / L01_at_10kHz)
        k = math.sqrt(1 - Lsc_at_10kHz / L01_at_10kHz)
        Lp = L01_at_10kHz * (1 + k) / 2
        print(f"turns_ratio: {turns_ratio}")
        print(f"k: {k}")
        print(f"Lp: {Lp}")
        print(f"cm1_at_10kHz: {cm1_at_10kHz}")
        print(f"cm2_resonances[0]['frequency']: {cm2_resonances[0]['frequency']}")
        print(f"cm3_resonances[0]['frequency']: {cm3_resonances[0]['frequency']}")
        print(f"cm4_resonances[0]['frequency']: {cm4_resonances[0]['frequency']}")
        print(f"cm5_resonances[0]['frequency']: {cm5_resonances[0]['frequency']}")
        print(f"cm6_resonances[0]['frequency']: {cm6_resonances[0]['frequency']}")

        def func(x):
            return [x[1] + x[2] - cm1_at_10kHz,
                    x[0] + math.pow(turns_ratio, 2) * x[1] - 1.0 / (Lp * pow(2 * math.pi * cm2_resonances[0]['frequency'], 2)),
                    x[0] + math.pow(turns_ratio, 2) * x[2] - 1.0 / (Lp * pow(2 * math.pi * cm3_resonances[0]['frequency'], 2))]
        [C1, C2, C3] = fsolve(func, [1, 1, 1])
        print(f"C1: {C1}")
        print(f"C2: {C2}")
        print(f"C3: {C3}")
        def func(x):
            return [x[0] + math.pow(turns_ratio, 2) * x[1] - C1,
                    x[4] + x[5] - C2,
                    x[2] - C3,
                    x[0] + x[2] + x[5] + math.pow(turns_ratio, 2) * x[3] + math.pow(1 + turns_ratio, 2) * x[4] - 1.0 / (Lsc * pow(2 * math.pi * cm4_resonances[0]['frequency'], 2)),
                    x[1] + x[3] + x[5] + - 1.0 / (Lsc * pow(2 * math.pi * cm5_resonances[0]['frequency'], 2)),
                    x[1] + x[2] + x[3] + x[4] + - 1.0 / (Lsc * pow(2 * math.pi * cm6_resonances[0]['frequency'], 2)),
                    ]
        [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6] = fsolve(func, [1, 1, 1, 1, 1, 1])
        print(f"gamma1: {gamma1}")
        print(f"gamma2: {gamma2}")
        print(f"gamma3: {gamma3}")
        print(f"gamma4: {gamma4}")
        print(f"gamma5: {gamma5}")
        print(f"gamma6: {gamma6}")


    def characterize_capacitance_advanced(self, allow_use_cache=False):
        #  According to https://sci-hub.st/https://ieeexplore.ieee.org/document/293449 and https://sci-hub.st/https://ieeexplore.ieee.org/document/382580

        data_Lsc_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_Lsc.csv"
        data_L01_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_L01.csv"
        data_L02_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_L02.csv"

        data_group_1_B_with_D_open_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_group_1_B_with_D_open.csv"
        data_group_1_B_with_D_short_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_group_1_B_with_D_short.csv"
        data_group_2_A_with_C_open_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_group_2_A_with_C_open.csv"
        data_group_2_A_with_C_short_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_group_2_A_with_C_short.csv"
        data_group_3_B_with_C_open_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_group_3_B_with_C_open.csv"
        data_group_3_B_with_C_short_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_group_3_B_with_C_short.csv"
        data_group_4_A_with_D_open_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_group_4_A_with_D_open.csv"
        data_group_4_A_with_D_short_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_group_4_A_with_D_short.csv"
        data_group_5_all_floating_open_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_group_5_all_floating_open.csv"
        data_group_5_all_floating_short_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_group_5_all_floating_short.csv"
        data_group_6_A_with_B_and_C_with_D_filepath  = self.output_path / f"{self.reference}_advanced_capacitance_characterization_data_group_6_A_with_B_and_C_with_D.csv"

        if allow_use_cache and os.path.exists(data_Lsc_filepath):
            data_Lsc = pandas.read_csv(data_Lsc_filepath)
        else:
            input("Place secondary in short circuit, setup primary up for measurement and press Enter to continue...")
            data_Lsc = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=40000000,
                number_of_measurement_cycles=2
            )
            data_Lsc.to_csv(data_Lsc_filepath, index=False)
            self.bode_100.plot(data_Lsc, "inductance", "Parallel capacitance")

        if allow_use_cache and os.path.exists(data_L01_filepath):
            data_L01 = pandas.read_csv(data_L01_filepath)
        else:
            input("Place secondary in open circuit, setup primary up for measurement and press Enter to continue...")
            data_L01 = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=40000000,
                number_of_measurement_cycles=2
            )
            data_L01.to_csv(data_L01_filepath, index=False)
            self.bode_100.plot(data_L01, "inductance", "Parallel capacitance")

        if allow_use_cache and os.path.exists(data_L02_filepath):
            data_L02 = pandas.read_csv(data_L02_filepath)
        else:
            input("Place primary in open circuit, setup secondary up for measurement and press Enter to continue...")
            data_L02 = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=40000000,
                number_of_measurement_cycles=2
            )
            data_L02.to_csv(data_L02_filepath, index=False)
            self.bode_100.plot(data_L02, "inductance", "Parallel capacitance")


        if allow_use_cache and os.path.exists(data_group_1_B_with_D_open_filepath):
            data_group_1_B_with_D_open = pandas.read_csv(data_group_1_B_with_D_open_filepath)
        else:
            input("Connect output of primary with output of secondary, leave secondary open, setup primary up for measurement and press Enter to continue...")
            data_group_1_B_with_D_open = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_group_1_B_with_D_open.to_csv(data_group_1_B_with_D_open_filepath, index=False)
            self.bode_100.plot_Z(data=data_group_1_B_with_D_open)

        if allow_use_cache and os.path.exists(data_group_1_B_with_D_short_filepath):
            data_group_1_B_with_D_short = pandas.read_csv(data_group_1_B_with_D_short_filepath)
        else:
            input("Connect output of primary with output of secondary, leave secondary short-circuited, setup primary up for measurement and press Enter to continue...")
            data_group_1_B_with_D_short = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_group_1_B_with_D_short.to_csv(data_group_1_B_with_D_short_filepath, index=False)
            self.bode_100.plot_Z(data=data_group_1_B_with_D_short)

        if allow_use_cache and os.path.exists(data_group_2_A_with_C_open_filepath):
            data_group_2_A_with_C_open = pandas.read_csv(data_group_2_A_with_C_open_filepath)
        else:
            input("Connect input of primary with input of secondary, leave secondary open, setup primary up for measurement and press Enter to continue...")
            data_group_2_A_with_C_open = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_group_2_A_with_C_open.to_csv(data_group_2_A_with_C_open_filepath, index=False)

        if allow_use_cache and os.path.exists(data_group_2_A_with_C_short_filepath):
            data_group_2_A_with_C_short = pandas.read_csv(data_group_2_A_with_C_short_filepath)
        else:
            input("Connect input of primary with input of secondary, leave secondary short-circuited, setup primary up for measurement and press Enter to continue...")
            data_group_2_A_with_C_short = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_group_2_A_with_C_short.to_csv(data_group_2_A_with_C_short_filepath, index=False)


        if allow_use_cache and os.path.exists(data_group_3_B_with_C_open_filepath):
            data_group_3_B_with_C_open = pandas.read_csv(data_group_3_B_with_C_open_filepath)
        else:
            input("Connect ouput of primary with input of secondary, leave secondary open, setup primary up for measurement and press Enter to continue...")
            data_group_3_B_with_C_open = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_group_3_B_with_C_open.to_csv(data_group_3_B_with_C_open_filepath, index=False)

        if allow_use_cache and os.path.exists(data_group_3_B_with_C_short_filepath):
            data_group_3_B_with_C_short = pandas.read_csv(data_group_3_B_with_C_short_filepath)
        else:
            input("Connect ouput of primary with input of secondary, leave secondary short-circuited, setup primary up for measurement and press Enter to continue...")
            data_group_3_B_with_C_short = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_group_3_B_with_C_short.to_csv(data_group_3_B_with_C_short_filepath, index=False)


        if allow_use_cache and os.path.exists(data_group_4_A_with_D_open_filepath):
            data_group_4_A_with_D_open = pandas.read_csv(data_group_4_A_with_D_open_filepath)
        else:
            input("Connect input of primary with output of secondary, leave secondary open, setup primary up for measurement and press Enter to continue...")
            data_group_4_A_with_D_open = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_group_4_A_with_D_open.to_csv(data_group_4_A_with_D_open_filepath, index=False)

        if allow_use_cache and os.path.exists(data_group_4_A_with_D_short_filepath):
            data_group_4_A_with_D_short = pandas.read_csv(data_group_4_A_with_D_short_filepath)
        else:
            input("Connect input of primary with output of secondary, leave secondary short-circuited, setup primary up for measurement and press Enter to continue...")
            data_group_4_A_with_D_short = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_group_4_A_with_D_short.to_csv(data_group_4_A_with_D_short_filepath, index=False)


        if allow_use_cache and os.path.exists(data_group_5_all_floating_open_filepath):
            data_group_5_all_floating_open = pandas.read_csv(data_group_5_all_floating_open_filepath)
        else:
            input("Leave secondary open, setup primary up for measurement and press Enter to continue...")
            data_group_5_all_floating_open = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_group_5_all_floating_open.to_csv(data_group_5_all_floating_open_filepath, index=False)

        if allow_use_cache and os.path.exists(data_group_5_all_floating_short_filepath):
            data_group_5_all_floating_short = pandas.read_csv(data_group_5_all_floating_short_filepath)
        else:
            input("Leave secondary short-circuited, setup primary up for measurement and press Enter to continue...")
            data_group_5_all_floating_short = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_group_5_all_floating_short.to_csv(data_group_5_all_floating_short_filepath, index=False)


        if allow_use_cache and os.path.exists(data_group_6_A_with_B_and_C_with_D_filepath):
            data_group_6_A_with_B_and_C_with_D = pandas.read_csv(data_group_6_A_with_B_and_C_with_D_filepath)
        else:
            input("Place secondary and primary in short circuit, connect each to one port and press Enter to continue...")
            data_group_6_A_with_B_and_C_with_D = self.bode_100.take_Cp_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            data_group_6_A_with_B_and_C_with_D.to_csv(data_group_6_A_with_B_and_C_with_D_filepath, index=False)


        _, lsc = self.extract_value_at_frequency(data_Lsc, 10000)
        _, L0 = self.extract_value_at_frequency(data_L01, 10000)
        _, L0prima = self.extract_value_at_frequency(data_L02, 10000)
        turns_ratio = math.sqrt(L0prima / L0)
        
        equations = []

        _, C33_at_10kHz = self.extract_value_at_frequency(data_group_6_A_with_B_and_C_with_D, 10000, parameter="capacitance")
        C33 = C33_at_10kHz

        data_group_1_B_with_D_open_resonances = self.detect_zero_crossing(data=data_group_1_B_with_D_open)
        data_group_1_B_with_D_short_resonances = self.detect_zero_crossing(data=data_group_1_B_with_D_short)
        assert len(data_group_1_B_with_D_open_resonances) >= 1
        # self.bode_100.plot_Z(data=data_group_1_B_with_D_short)
        # assert len(data_group_1_B_with_D_short_resonances) >= 1

        if len(data_group_1_B_with_D_open_resonances) > 0:
            assert data_group_1_B_with_D_open_resonances[0]['type'] == "local maximum"
            f2 = data_group_1_B_with_D_open_resonances[0]['frequency']
            C1_plus_C2 = 1.0 / (L0 * pow(2 * math.pi * f2, 2))
            equations.append(f"C11 + math.pow({turns_ratio}, 2) * C22 + 2 * {turns_ratio} * C12 - {C1_plus_C2}")

        if len(data_group_1_B_with_D_short_resonances) > 0:
            assert data_group_1_B_with_D_short_resonances[0]['type'] == "local maximum"
            f3 = data_group_1_B_with_D_short_resonances[0]['frequency']
            C1_plus_C3 = 1.0 / (lsc * pow(2 * math.pi * f3, 2))
            equations.append(f"C11 - {C1_plus_C3}")


        if len(data_group_1_B_with_D_open_resonances) > 1:
            assert data_group_1_B_with_D_open_resonances[1]['type'] == "local minimum"
            f4 = data_group_1_B_with_D_open_resonances[1]['frequency']
            C2_plus_C3 = 1.0 / (lsc * pow(2 * math.pi * f4, 2))
            equations.append(f"math.pow({turns_ratio}, 2) * C22 - {C2_plus_C3}")


        data_group_2_A_with_C_open_resonances = self.detect_zero_crossing(data=data_group_2_A_with_C_open)
        data_group_2_A_with_C_short_resonances = self.detect_zero_crossing(data=data_group_2_A_with_C_short)


        if len(data_group_2_A_with_C_open_resonances) > 0:
            assert data_group_2_A_with_C_open_resonances[0]['type'] == "local maximum"
            f2 = data_group_2_A_with_C_open_resonances[0]['frequency']
            C1_plus_C2 = 1.0 / (L0 * pow(2 * math.pi * f2, 2))
            equations.append(f"C11 + {C33} + 2 * C13 + math.pow({turns_ratio}, 2) * (C22 + {C33} - 2 * C23) + 2 * {turns_ratio} * (C12 - {C33} - C13 + C23) - {C1_plus_C2}")


        if len(data_group_2_A_with_C_short_resonances) > 0:
            assert data_group_2_A_with_C_short_resonances[0]['type'] == "local maximum"
            f3 = data_group_2_A_with_C_short_resonances[0]['frequency']
            C1_plus_C3 = 1.0 / (lsc * pow(2 * math.pi * f3, 2))
            equations.append(f"C11 + {C33} + 2 * C13 - {C1_plus_C3}")

        if len(data_group_2_A_with_C_open_resonances) > 1:
            assert data_group_2_A_with_C_open_resonances[1]['type'] == "local minimum"
            f4 = data_group_2_A_with_C_open_resonances[1]['frequency']
            C2_plus_C3 = 1.0 / (lsc * pow(2 * math.pi * f4, 2))
            equations.append(f"math.pow({turns_ratio}, 2) * (C22 + {C33} - 2 * C23) - {C2_plus_C3}")

        data_group_3_B_with_C_open_resonances = self.detect_zero_crossing(data=data_group_3_B_with_C_open)
        data_group_3_B_with_C_short_resonances = self.detect_zero_crossing(data=data_group_3_B_with_C_short)
        if len(data_group_3_B_with_C_open_resonances) > 0:
            assert data_group_3_B_with_C_open_resonances[0]['type'] == "local maximum"
            f2 = data_group_3_B_with_C_open_resonances[0]['frequency']
            C1_plus_C2 = 1.0 / (L0 * pow(2 * math.pi * f2, 2))
            equations.append(f"C11 + math.pow({turns_ratio}, 2) * (C22 + {C33} - 2 * C23) + 2 * {turns_ratio} * (C12 - C13) - {C1_plus_C2}")

        if len(data_group_3_B_with_C_short_resonances) > 0:
            assert data_group_3_B_with_C_short_resonances[0]['type'] == "local maximum"
            f3 = data_group_3_B_with_C_short_resonances[0]['frequency']
            C1_plus_C3 = 1.0 / (lsc * pow(2 * math.pi * f3, 2))
            equations.append(f"C11 - {C1_plus_C3}")

        if len(data_group_3_B_with_C_short_resonances) > 1:
            assert data_group_3_B_with_C_open_resonances[1]['type'] == "local minimum"
            f4 = data_group_3_B_with_C_open_resonances[1]['frequency']
            C2_plus_C3 = 1.0 / (lsc * pow(2 * math.pi * f4, 2))
            equations.append(f"math.pow({turns_ratio}, 2) * (C22 + {C33} - 2 * C23) - {C2_plus_C3}")

        data_group_4_A_with_D_open_resonances = self.detect_zero_crossing(data=data_group_4_A_with_D_open)
        data_group_4_A_with_D_short_resonances = self.detect_zero_crossing(data=data_group_4_A_with_D_short)

        if len(data_group_4_A_with_D_open_resonances) > 0:
            assert data_group_4_A_with_D_open_resonances[0]['type'] == "local maximum"
            f2 = data_group_4_A_with_D_open_resonances[0]['frequency']
            C1_plus_C2 = 1.0 / (L0 * pow(2 * math.pi * f2, 2))
            equations.append(f"C11 + {C33} + 2 * C13 + math.pow({turns_ratio}, 2) * C22 + 2 * {turns_ratio} * (C12 + C23) - {C1_plus_C2}")

        if len(data_group_4_A_with_D_short_resonances) > 0:
            assert data_group_4_A_with_D_short_resonances[0]['type'] == "local maximum"
            f3 = data_group_4_A_with_D_short_resonances[0]['frequency']
            C1_plus_C3 = 1.0 / (lsc * pow(2 * math.pi * f3, 2))
            equations.append(f"C11 + {C33} + 2 * C13 - {C1_plus_C3}")

        if len(data_group_4_A_with_D_open_resonances) > 1:
            assert data_group_4_A_with_D_open_resonances[1]['type'] == "local minimum"
            f4 = data_group_4_A_with_D_open_resonances[1]['frequency']
            C2_plus_C3 = 1.0 / (lsc * pow(2 * math.pi * f4, 2))
            equations.append(f"math.pow({turns_ratio}, 2) * C22 - {C2_plus_C3}")

        data_group_5_all_floating_open_resonances = self.detect_zero_crossing(data=data_group_5_all_floating_open)
        data_group_5_all_floating_short_resonances = self.detect_zero_crossing(data=data_group_5_all_floating_short)

        if len(data_group_5_all_floating_open_resonances) > 0:
            assert data_group_5_all_floating_open_resonances[0]['type'] == "local maximum"
            f2 = data_group_5_all_floating_open_resonances[0]['frequency']
            C1_plus_C2 = 1.0 / (L0 * pow(2 * math.pi * f2, 2))
            equations.append(f"C11 - math.pow(C13, 2) / {C33} + math.pow({turns_ratio}, 2) * (C22 - math.pow(C23, 2) / {C33}) + 2 * {turns_ratio} * (C12 - C13 * C23 / {C33}) - {C1_plus_C2}")

        if len(data_group_5_all_floating_short_resonances) > 0:
            assert data_group_5_all_floating_short_resonances[0]['type'] == "local maximum"
            f3 = data_group_5_all_floating_short_resonances[0]['frequency']
            C1_plus_C3 = 1.0 / (lsc * pow(2 * math.pi * f3, 2))
            equations.append(f"C11 - math.pow(C13, 2) / {C33} - {C1_plus_C3}")

        if len(data_group_5_all_floating_open_resonances) > 1:
            assert data_group_5_all_floating_open_resonances[1]['type'] == "local minimum"
            f4 = data_group_5_all_floating_open_resonances[1]['frequency']
            C2_plus_C3 = 1.0 / (lsc * pow(2 * math.pi * f4, 2))
            equations.append(f"math.pow({turns_ratio}, 2) * (C22 - math.pow(C23, 2) / {C33}) - {C2_plus_C3}")


        print(equations)
        def func(variables):
            (C11, C12, C13, C22, C23) = variables
            res = []
            for eq in equations:
                res.append(eval(eq))
            return res

        result = least_squares(func, [0.1e-9, 0.1e-9, 0.1e-9, 0.1e-9, 0.1e-9], loss='cauchy', f_scale=0.1, ftol=1e-05, xtol=1e-05)
        print(result)
        [C11, C12, C13, C22, C23] = result.x


        def func(variables):
            (C1, C2, C3) = variables
            return [C1 + C3 - C11,
                    C2 + C3 - math.pow(turns_ratio, 2) * C22,
                    C1 + C2 - C11 - math.pow(turns_ratio, 2) * C22 - 2 * turns_ratio * C12]
        [C1, C2, C3] = fsolve(func, [1, 1, 1])
        print(f"C1: {C1}")
        print(f"C2: {C2}")
        print(f"C3: {C3}")

    def characterize_ac_resistance(self, allow_use_cache=False):

        data_1_filepath  = self.output_path / f"{self.reference}_ac_resistance_characterization_data_1.csv"
        data_Z1_filepath  = self.output_path / f"{self.reference}_ac_resistance_characterization_data_Z1.csv"
        data_Z2_filepath  = self.output_path / f"{self.reference}_ac_resistance_characterization_data_Z2.csv"
        data_2_filepath  = self.output_path / f"{self.reference}_ac_resistance_characterization_data_2.csv"

        if allow_use_cache and os.path.exists(data_1_filepath):
            data_1 = pandas.read_csv(data_1_filepath)
        else:
            input("Place winding with first gap combination...")
            data_1 = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=100000,
                number_of_measurement_cycles=2
            )
            data_1.to_csv(data_1_filepath, index=False)

        if allow_use_cache and os.path.exists(data_Z1_filepath):
            data_Z1 = pandas.read_csv(data_Z1_filepath)
        else:
            data_Z1 = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            self.bode_100.plot_Z(data=data_Z1)
            data_Z1.to_csv(data_Z1_filepath, index=False)

        resonances = self.detect_zero_crossing(data=data_Z1)
        print(resonances)

        _, Lmag_10k = self.extract_value_at_frequency(data_1, 100)
        _, R1_100 = self.extract_value_at_frequency(data_1, 100, "resistance")
        print("R1_100")
        print(R1_100)

        w_res = 2 * math.pi * resonances[0]["frequency"]
        Cp = 1. / (w_res**2 * Lmag_10k)
        print(Cp)
        print(Lmag_10k)
        data_1["Rcw"] = [0] * len(data_1.index)
        for index, row in data_1.iterrows():
            w = 2 * math.pi * row["frequency"]
            Rm = row["resistance"]
            Rcw = 1. / (2 * Cp**2 * w**2 * Rm) * (1 - math.sqrt(2 * Lmag_10k * Rm * Cp**2 * w**3 - 2 * Rm * Cp * w + 1) * math.sqrt(-2 * Lmag_10k * Rm * Cp**2 * w**3 + 2 * Rm * Cp * w + 1))
            data_1.loc[index, "Rcw"] = Rcw 

        if allow_use_cache and os.path.exists(data_2_filepath):
            data_2 = pandas.read_csv(data_2_filepath)
        else:
            input("Place winding with second gap combination...")
            data_2 = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=10000,
                stop_frequency=100000,
                number_of_measurement_cycles=2
            )
            data_2.to_csv(data_2_filepath, index=False)

        if allow_use_cache and os.path.exists(data_Z2_filepath):
            data_Z2 = pandas.read_csv(data_Z2_filepath)
        else:
            data_Z2 = self.bode_100.take_Z_phase_measurement(
                number_of_measurement_cycles=2,
                start_frequency=10000,
                stop_frequency=40000000
            )
            self.bode_100.plot_Z(data=data_Z2)
            data_Z2.to_csv(data_Z2_filepath, index=False)

        resonances = self.detect_zero_crossing(data=data_Z2)
        print(resonances)

        _, Lmag_2_10k = self.extract_value_at_frequency(data_2, 100)
        _, R2_100 = self.extract_value_at_frequency(data_2, 100, "resistance")
        print("R2_100")
        print(R2_100)


        w_res = 2 * math.pi * resonances[0]["frequency"]
        Cp = 1. / (w_res**2 * Lmag_2_10k)
        data_2["Rcw"] = [0] * len(data_2.index)
        for index, row in data_2.iterrows():
            w = 2 * math.pi * row["frequency"]
            Rm = row["resistance"]
            Rcw = 1. / (2 * Cp**2 * w**2 * Rm) * (1 - math.sqrt(2 * Lmag_10k * Rm * Cp**2 * w**3 - 2 * Rm * Cp * w + 1) * math.sqrt(-2 * Lmag_10k * Rm * Cp**2 * w**3 + 2 * Rm * Cp * w + 1))
            data_2.loc[index, "Rcw"] = Rcw 

        data_1["Rw"] = data_1["Rcw"] - (2 * math.pi * data_1["frequency"] * Lmag_10k)**2 / (((2 * math.pi * data_1["frequency"] * Lmag_10k)**2 - (2 * math.pi * data_1["frequency"] * Lmag_2_10k)**2) / (data_1["Rcw"] - data_2["Rcw"]))
        data_1["Rc"] = data_1["Rcw"] - data_1["Rw"]
        print(data_1)
        print(data_2)
        self.bode_100.plot(
            data=data_1,
            column="Rw",
            label="Rw",
        )




if __name__ == "__main__":
    # characterizer = MagneticCharacterizer("750315213")  # Small
    # characterizer = MagneticCharacterizer("750341867")  # Large
    # characterizer = MagneticCharacterizer("Custom_Two_Layers")  # Large
    # characterizer = MagneticCharacterizer("Custom_Inductor")  # Large
    # characterizer = MagneticCharacterizer("Flyback_0")  # Large
    # characterizer = MagneticCharacterizer("Flyback_1")  # Large
    characterizer = MagneticCharacterizer("Flyback_2")  # Large
    # characterizer.characterize_inductance_basic(True)
    # characterizer.characterize_inductance_medium(True)
    # characterizer.characterize_inductance_advanced(True)
    # characterizer.extract_resonances(False)
    # characterizer.characterize_capacitance_advanced(True)
    # characterizer.characterize_capacitance_medium(True)
    # characterizer.characterize_capacitance_medium(False)
    characterizer.characterize_ac_resistance(True)
