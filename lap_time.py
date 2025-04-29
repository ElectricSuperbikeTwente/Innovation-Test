import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QtAgg')
import pandas as pd
import math

circuit_data = pd.read_excel('aragon.xlsx')
soc_data = pd.read_excel('soc_curve.xlsx')
pcm_3690 = pd.read_excel('PCM ATPC 36 90 - 10.xlsx')


## Input Variables

# Gear info
first_gear_ratio = 1.5
second_gear_ratio = 1.5
rear_chain_ratio = 3.79/1.5
transmission_efficiency_gearbox = 0.98*0.98
transmission_efficiency_chain = 0.98

# Motor
motor_max_rpm = 6000
motor_max_power = 40 # input in kw, this will decrease with the efficiency
motor_max_torque = 100 # in NM
motor_efficiency = 0.93

# inverter efficiency
inverter_efficiency = 0.96

# Simulation
simulation_time_step = 0.01 # time in s

# Wheels
wheel_diameter = 0.603 # in M

# BDU
bdu_current_limit = 200

# Battery Pack
bp_resistance = 0.004590563/7*30 # in ohms (0.068 from datasheet) should be updated from damstra tests
bp_start_voltage = 126
bp_end_voltage = 95
bp_voltagedrop = 0.1/12.5 * 30/14 # voltage drop in V/A for the whole pack
bp_cell_weight = 48.72 # Weight of the cells in the BP
bp_cell_heat_capacity = 1300 # in J/(kg-K)
bp_cooling_material_1_weight = 0.723 # Weight of the copper pieces in the BP
bp_cooling_material_1_heat_capacity = 385 # in J/(kg-K)
bp_cooling_material_2_weight = 0.05 # Weight of the air in the BP
bp_cooling_material_2_heat_capacity = 1006 # in J/(kg-K)
bp_cooling_material_3_weight = 0
bp_cooling_material_3_heat_capacity = 921.096 # in J/(kg-K)
bp_cooling_material_4_weight = 0
bp_cooling_material_4_heat_capacity = 921.096 # in J/(kg-K)
bp_cooling_material_5_weight = 0
bp_cooling_material_5_heat_capacity = 921.096 # in J/(kg-K)
bp_cooling_pcm_3690_weight = 0
bp_cell_parallel = 14
bp_cell_series = 30

# soc curve
soc_discharge_curve_drop = 0.06
soc_amperage = 1.32
soc_total_time = 304*60 # in seconds

# Mass
mass_motorcycle = 170
mass_person = 73
mass_inertia_equivalent = 7.5

# Resistance
rolling_resistance_coef = 0.02
frontal_area = 0.6
drag_coefficient = 0.61

# Circuit conditions
air_pressure = 960
temperature = 18 # in degrees celsius

# Brakes
braking_deceleration = 1.1 #in g

# laps
circuit_laps = 7







## Calculated Variables
wheel_circumference = math.pi * wheel_diameter
mass_total = mass_motorcycle + mass_person + mass_inertia_equivalent
air_density = (air_pressure*100)/(287.058*(temperature+273.15))

bp_cooled_weight = bp_cell_weight + bp_cooling_material_1_weight + bp_cooling_material_2_weight + bp_cooling_material_3_weight + bp_cooling_material_4_weight + bp_cooling_material_5_weight

transmission_efficiency_total = transmission_efficiency_chain*transmission_efficiency_gearbox

bp_heat_capacity = (bp_cell_heat_capacity*bp_cell_weight + bp_cooling_material_1_heat_capacity*bp_cooling_material_1_weight + bp_cooling_material_2_heat_capacity*bp_cooling_material_2_weight + bp_cooling_material_3_heat_capacity*bp_cooling_material_3_weight + bp_cooling_material_4_heat_capacity*bp_cooling_material_4_weight + bp_cooling_material_5_heat_capacity*bp_cooling_material_5_weight)/(bp_cooled_weight)

soc_curve_voltage = soc_data['voltage']+soc_discharge_curve_drop
soc_curve_amperage = np.ones(len(soc_curve_voltage))*soc_amperage
soc_curve_timestep = soc_total_time/len(soc_curve_voltage)
soc_curve_w = np.multiply(soc_curve_voltage,soc_curve_amperage)
soc_curve_w_used = list()

temp_cummulative_power = np.multiply(soc_curve_w,soc_curve_timestep)
for i in range(0,len(soc_curve_voltage)):
    soc_curve_w_used.append(np.sum(temp_cummulative_power[0:(i+1)]))

circuit_length = circuit_data['Distance'][len(circuit_data['Distance'])-1]

bp_total_cooling_array_temperature = list()
bp_total_cooling_array_total_heat = list()
temp_total_heat = 0

for i in np.linspace(temperature, temperature+50,501):
    bp_total_cooling_array_temperature.append(i)
    temp_total_heat += round(np.interp(i,pcm_3690['temperature'],(pcm_3690['heat capacity']*bp_cooling_pcm_3690_weight*1000 + bp_heat_capacity*bp_cooled_weight)),2)*0.1
    bp_total_cooling_array_total_heat.append(temp_total_heat)
    




## Gearing Info
first_gear_total_ratio = first_gear_ratio*rear_chain_ratio
first_gear_top_speed = motor_max_rpm*wheel_circumference/60/first_gear_total_ratio*3.6
first_gear_max_torque = motor_max_torque*first_gear_total_ratio

second_gear_total_ratio = second_gear_ratio*rear_chain_ratio
second_gear_top_speed = motor_max_rpm*wheel_circumference/60/second_gear_total_ratio*3.6
second_gear_max_torque = motor_max_torque*second_gear_total_ratio




# Calculating the characteristics for each speed
char_motor_rpm = list()
char_speed_kmh = list(np.round(np.linspace(0,second_gear_top_speed,2000),2))
char_speed = np.multiply(char_speed_kmh,3.6)
char_wheel_rpm = list()
char_wheel_torque = list()
char_gear = list()
char_motor_torque = list()
char_motor_output_power = list()
char_motor_input_power = list()

for speed_kmh in char_speed_kmh:
    if speed_kmh > first_gear_top_speed:
        char_gear.append(2)
    else:
        char_gear.append(1)

    char_wheel_rpm.append(speed_kmh/3.6/wheel_circumference*60)

    if char_gear[len(char_gear)-1] == 1:
        char_motor_rpm.append(char_wheel_rpm[len(char_wheel_rpm)-1]*first_gear_total_ratio)

        if (first_gear_max_torque*char_wheel_rpm[len(char_wheel_rpm)-1]/60*2*math.pi/1000) > motor_max_power*motor_efficiency:
            char_wheel_torque.append(motor_max_power*motor_efficiency/(char_wheel_rpm[len(char_wheel_rpm)-1]/60*2*math.pi)*1000*transmission_efficiency_total)
        else:
            char_wheel_torque.append(first_gear_max_torque*transmission_efficiency_total)

        char_motor_torque.append(char_wheel_torque[len(char_wheel_torque)-1]/(first_gear_total_ratio)/transmission_efficiency_total)
    else:
        char_motor_rpm.append(char_wheel_rpm[len(char_wheel_rpm)-1]*second_gear_total_ratio)

        if (second_gear_max_torque*char_wheel_rpm[len(char_wheel_rpm)-1]/60*2*math.pi/1000) > motor_max_power*motor_efficiency:
            char_wheel_torque.append(motor_max_power*motor_efficiency/(char_wheel_rpm[len(char_wheel_rpm)-1]/60*2*math.pi)*1000*transmission_efficiency_total)
        else:
            char_wheel_torque.append(second_gear_max_torque*transmission_efficiency_total)

        char_motor_torque.append(char_wheel_torque[len(char_wheel_torque)-1]/(second_gear_total_ratio)/transmission_efficiency_total)

    char_motor_output_power.append(2*math.pi*char_motor_rpm[len(char_motor_rpm)-1]*char_motor_torque[len(char_motor_torque)-1]/60)
    char_motor_input_power.append(char_motor_output_power[len(char_motor_output_power)-1]/motor_efficiency)






# Variables for the simulation
finishedCircuit = False
current_time = 0
current_acceleration = 0
current_speed = 0
current_speed_kmh = 0
current_distance = 0
current_lap_distance = 0
current_force_wheel = 0
current_force_rolling_resistance = 0
current_force_drag_resistance = 0
current_force_total = 0
current_wheel_torque = 0
current_max_speed = 0
current_gear = 0
current_gearing = 0
current_motor_rpm = 0
current_motor_torque = 0
current_motor_output_power = 0
current_motor_input_power = 0
current_state = 1 ##state 0:idle, state 1: throttle, state 2: brake, state 3: maintain speed
current_motor_heat_kw = 0
current_gearbox_heat_kw = 0
current_chain_heat_kw = 0
current_bp_heat_kw = 0
current_inverter_heat_kw = 0
current_bp_power_kw = 0
current_bp_voltage_unloaded = 0
current_bp_voltage_loaded = 0
current_bp_current = 0
current_bp_temperature = temperature
current_bp_w_used = 0
current_bp_total_heat = 0
current_lap = 1

current_lap_time = 0

time = list()
acceleration = list()
speed = list()
speed_kmh = list()
distance = list()
lap_distance = list()
force_wheel = list()
force_rolling_resistance = list()
force_drag_resistance = list()
force_total = list()
wheel_torque = list()
max_speed = list()
gear = list()
motor_rpm = list()
motor_torque = list()
motor_output_power = list()
motor_input_power = list()
state = list()
motor_heat_kw = list()
gearbox_heat_kw = list()
chain_heat_kw = list()
bp_heat_kw = list()
inverter_heat_kw = list()
bp_power_kw = list()
bp_voltage_unloaded = list()
bp_voltage_loaded = list()
bp_current = list()
bp_temperature = list()
bp_w_used = list()
bp_total_heat = list()
lap = list()

lap_time = list()

temp_bp_max_kw = (bp_start_voltage-bp_voltagedrop*bdu_current_limit)*bdu_current_limit

# The simulation for the selected track
while finishedCircuit == False:
        # Check the current max speed
    for k in range(0,len(circuit_data['Distance'])):
        if current_lap_distance >= circuit_data['Distance'][k]:
            current_max_speed = circuit_data['Max Speed'][k]

    # Check if there is a need for some braking
    for j in range(0,len(circuit_data['Distance'])):
        if current_lap_distance <= circuit_data['Distance'][j]:
            temp_data_max_speed = circuit_data['Max Speed'][j]
            temp_data_distance = circuit_data['Distance'][j]

            temp_data_delta_distance = temp_data_distance - current_lap_distance
            
            # check if the datapoint is coming up
            if (temp_data_delta_distance > 0):

                # Calculate a maximum speed for this datapoint based on the braking distance to the maximum speed at this point
                temp_data_new_speed_ms = np.sqrt((temp_data_max_speed/3.6)**2 + 2*braking_deceleration*9.81*temp_data_delta_distance)
                temp_data_new_speed = temp_data_new_speed_ms*3.6

                # Check if the calculated speed for this points is lower than the maximum section speed or the braking speed for a different point
                if (temp_data_new_speed < current_max_speed):
                    current_max_speed = temp_data_new_speed
                

    # Check if we have to throttle
    if (current_speed_kmh < current_max_speed):
        current_state = 1

    # Check if we are at max speed, at which it can continue cruising
    if (current_speed_kmh >= first_gear_top_speed and current_speed_kmh >= second_gear_top_speed):
        current_state = 3

    # Check if we have to brake
    if (current_speed_kmh >= current_max_speed):
        current_state = 2

    # Check if we are in the cruise zone
    speed_comparison_delta = 0.1
    if (current_speed_kmh <= current_max_speed+speed_comparison_delta and current_speed_kmh >= current_max_speed-speed_comparison_delta):
        # We have to maintain speed
        current_state = 3

   
    # Calculate the acceleration with the resistance and wheel torque which is influenced by the current state
    if (current_state == 1):
        current_wheel_torque = np.interp(current_speed_kmh,char_speed_kmh,char_wheel_torque)

        # Power limiter
        temp_power = 2*(current_speed*first_gear_total_ratio/wheel_circumference)*current_wheel_torque
        if (temp_power > temp_bp_max_kw):
            temp_scaling_factor_state1 = (temp_bp_max_kw/temp_power)
            current_wheel_torque = current_wheel_torque*temp_scaling_factor_state1
        else:
            temp_scaling_factor_state1 = 1

        current_force_wheel = current_wheel_torque/(0.5*wheel_diameter)
        current_force_rolling_resistance = -9.81*rolling_resistance_coef*mass_total
        current_force_drag_resistance = -0.5*current_speed**2*frontal_area*drag_coefficient*air_density
        current_force_total = current_force_wheel + current_force_rolling_resistance + current_force_drag_resistance

        current_acceleration = current_force_total/mass_total

    if (current_state == 2):
        current_acceleration = -braking_deceleration*9.81

    if (current_state == 3):
        current_force_rolling_resistance = -9.81*rolling_resistance_coef*mass_total
        current_force_drag_resistance = -0.5*current_speed**2*frontal_area*drag_coefficient*air_density
        current_force_wheel = -(current_force_rolling_resistance + current_force_drag_resistance)
        current_wheel_torque = (0.5*wheel_diameter)*current_force_wheel

        # Power limiter
        temp_power = 2*(current_speed*first_gear_total_ratio/wheel_circumference)*current_wheel_torque
        if (temp_power > temp_bp_max_kw):
            current_wheel_torque = current_wheel_torque*(temp_bp_max_kw/temp_power)

        current_acceleration = 0
    
    # update the current speed
    current_speed = current_acceleration*simulation_time_step + current_speed
    current_speed_kmh = current_speed*3.6
    
    # update the current distance
    current_distance = current_distance + current_speed*simulation_time_step
    current_lap_distance = current_distance - circuit_length*(current_lap-1)

    # find out which gear we are in
    current_gear = np.interp(current_speed_kmh,char_speed_kmh,char_gear)
    if current_gear == 1:
        current_motor_rpm = current_speed*first_gear_total_ratio/wheel_circumference*60
        current_gearing = first_gear_total_ratio
    else:
        current_motor_rpm = current_speed*second_gear_total_ratio/wheel_circumference*60
        current_gearing = second_gear_total_ratio
        

    # Calculate the torque based on the current state
    if (current_state == 1):
        current_motor_torque = np.interp(current_speed_kmh,char_speed_kmh,char_motor_torque)
        current_motor_torque = current_motor_torque*temp_scaling_factor_state1

    if (current_state == 2):
        current_motor_torque = 0

    if (current_state == 3):
        current_motor_torque = current_wheel_torque/current_gearing/transmission_efficiency_total

        
    # Calculate motor power and heat
    current_motor_output_power = 2*math.pi*current_motor_rpm/60*current_motor_torque
    current_motor_input_power = current_motor_output_power/motor_efficiency
    current_motor_heat_kw = (current_motor_input_power-current_motor_output_power)/1000

    # Calculate transmission heat
    current_gearbox_heat_kw = (1-transmission_efficiency_gearbox)*current_motor_output_power/1000
    current_chain_heat_kw = (1-transmission_efficiency_chain)*current_motor_output_power*transmission_efficiency_gearbox/1000
    
    # Calculate inverter heat
    current_inverter_heat_kw = (current_motor_input_power/inverter_efficiency-current_motor_input_power)/1000
    
    # Calculate BP heat with soc
    current_bp_power_kw = current_motor_input_power/inverter_efficiency/1000
    current_bp_w_used = current_bp_w_used + current_bp_power_kw*simulation_time_step*1000
    current_bp_voltage_unloaded = np.interp(current_bp_w_used/(bp_cell_series*bp_cell_parallel),soc_curve_w_used,soc_curve_voltage)*bp_cell_series
    current_bp_voltage_loaded = current_bp_voltage_unloaded-current_bp_current*bp_voltagedrop
    current_bp_current = current_bp_power_kw*1000/current_bp_voltage_loaded
    current_bp_heat_kw = current_bp_current**2*bp_resistance/1000
    current_bp_total_heat += current_bp_heat_kw*simulation_time_step*1000
    temp_bp_max_kw = (current_bp_voltage_unloaded-bp_voltagedrop*bdu_current_limit)*bdu_current_limit

    current_bp_temperature = round(np.interp(current_bp_total_heat,bp_total_cooling_array_total_heat,bp_total_cooling_array_temperature),2)
    # current_bp_temperature = current_bp_temperature + (current_bp_heat_kw*simulation_time_step*1000)/(bp_heat_capacity*(bp_cooled_weight))
    

    # Saving the data to the lists
    time.append(current_time)
    acceleration.append(current_acceleration)
    speed.append(current_speed)
    speed_kmh.append(current_speed_kmh)
    distance.append(current_distance)
    lap_distance.append(current_lap_distance)
    force_wheel.append(current_force_wheel)
    force_drag_resistance.append(current_force_drag_resistance)
    force_rolling_resistance.append(current_force_rolling_resistance)
    force_total.append(current_force_total)
    wheel_torque.append(current_wheel_torque)
    max_speed.append(current_max_speed)
    gear.append(current_gear)
    motor_rpm.append(current_motor_rpm)
    motor_torque.append(current_motor_torque)
    motor_output_power.append(current_motor_output_power)
    motor_input_power.append(current_motor_input_power)
    state.append(current_state)
    motor_heat_kw.append(current_motor_heat_kw)
    gearbox_heat_kw.append(current_gearbox_heat_kw)
    chain_heat_kw.append(current_chain_heat_kw)
    bp_heat_kw.append(current_bp_heat_kw)
    inverter_heat_kw.append(current_inverter_heat_kw)
    bp_power_kw.append(current_bp_power_kw)
    bp_voltage_loaded.append(current_bp_voltage_loaded)
    bp_voltage_unloaded.append(current_bp_voltage_unloaded)
    bp_current.append(current_bp_current)
    bp_temperature.append(current_bp_temperature)
    bp_w_used.append(current_bp_w_used)
    bp_total_heat.append(current_bp_total_heat)

    # Check if we have made a lap
    if current_lap_distance >= circuit_length:
        lap_time.append(round(current_time-current_lap_time,2))
        current_lap_time = current_time
        # check if we need to make for laps
        if (current_lap < circuit_laps):
            current_lap += 1
            current_lap_distance = current_distance - circuit_length
        else:
            finishedCircuit = True
    else:
        # update the current time
        current_time = round(current_time + simulation_time_step,4)


wheel_power = np.multiply(motor_output_power,transmission_efficiency_total)
total_heat_kw = np.sum([chain_heat_kw,gearbox_heat_kw,bp_heat_kw,inverter_heat_kw,motor_heat_kw], axis=0)
efficiency = np.divide(wheel_power,np.sum([np.multiply(bp_power_kw,1000),np.multiply(bp_heat_kw,1000),np.ones(len(bp_heat_kw))*0.1], axis=0))
efficiency[(efficiency<= 0.01)] = float('NaN')


print(lap_time)



##plotting the motor dynamic results
fig1, axs1 = plt.subplots(2,2)
fig1.suptitle(' Motor Dynamics ', fontsize=30)
axs1[0,0].plot(char_speed_kmh,char_wheel_torque)
axs1[0,0].set_title('Wheel Torque vs Speed')
axs1[0,0].set_xlabel('Speed (km/h)')
axs1[0,0].set_ylabel('Torque (nm)')
axs_gear0 = axs1[0,0].twinx()
axs_gear0.plot(char_speed_kmh,char_gear,color='black', linestyle='dashed', linewidth=1)

axs1[0,1].plot(char_speed_kmh,char_motor_output_power)
axs1[0,1].set_title('Speed vs output power & torque')
axs1[0,1].set_xlabel('Speed (km/h)')
axs1[0,1].set_ylabel('Power (kW)')
axs_torque0 = axs1[0,1].twinx()
axs_torque0.plot(char_speed_kmh,char_motor_torque,color='black')
axs_torque0.set_ylabel('Torque (NM)')


axs1[1,0].plot(char_motor_rpm,char_motor_output_power, linestyle='none', marker=".", markersize ='0.5')
axs1[1,0].set_title('RPM vs output power & torque')
axs1[1,0].set_xlabel('Speed (RPM)')
axs1[1,0].set_ylabel('Power (kW)')
axs_torque0 = axs1[1,0].twinx()
axs_torque0.plot(char_motor_rpm,char_motor_torque,color='black', linestyle='none', marker=".", markersize ='0.5')
axs_torque0.set_ylabel('Torque (NM)')

axs1[1,1].plot(bp_total_cooling_array_temperature,bp_total_cooling_array_total_heat, linestyle='none', marker=".", markersize ='0.5')
axs1[1,1].set_title('Temperature vs total heat capacity')
axs1[1,1].set_xlabel('Temperature')
axs1[1,1].set_ylabel('Total Heat (J)')

figManager1 = plt.get_current_fig_manager()
figManager1.window.showMaximized()





##plotting the simulation results
fig, axs = plt.subplots(2,2)
fig.suptitle(' Race Simulation in ' + str(round(current_time,2)) + ' seconds', fontsize=30)
axs[0,0].plot(time,bp_current)
axs[0,0].set_title('BP Current (A) vs time (s)')
axs[0,0].set_xlabel('Time (s)')
axs[0,0].set_ylabel('Current (A)')
axs[0,0].set_ylim([-0.05*max(bp_current),1.05*max(bp_current)])
axs[0,0].legend(['Current'],loc='lower right')
# axs_gear1 = axs[0,0].twinx()
# axs_gear1.set_ylim([1.9,3])
# axs_gear1.plot(time,gear,color='orange', linestyle='dashed', linewidth=1)

axs[0,1].plot(time,efficiency)
axs[0,1].set_title('Efficiency vs time')
axs[0,1].set_xlabel('Time (s)')
axs[0,1].set_ylabel('Efficiency (%)')

axs[1,0].plot(time,motor_rpm)
axs[1,0].set_title('RPM & Torque vs time')
axs[1,0].set_xlabel('Time (s)')
axs[1,0].set_ylabel('RPM')
axs_torque1 = axs[1,0].twinx()
axs_torque1.plot(time,motor_torque,color='black', linestyle='none', marker=".", markersize ='0.5')
axs_torque1.set_ylabel('Torque (NM)')

axs[1,1].plot(time,bp_power_kw)
axs[1,1].set_title('Battery power (' + str(np.round(np.sum(np.multiply(bp_power_kw,simulation_time_step))/3600,3)/circuit_laps) + ' kWh/lap) (' + str(np.round(np.sum(np.multiply(bp_power_kw,simulation_time_step))/3600,3)) +  'kWh total) (' + str(np.round(np.average(bp_power_kw),3)) + ' KW average) vs Time')
axs[1,1].set_xlabel('Time (s)')
axs[1,1].set_ylabel('Power (kW)')

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()





fig, axs2 = plt.subplots(2,2)
fig.suptitle(' Race Simulation in ' + str(round(current_time,2)) + ' seconds', fontsize=30)
axs2[0,0].plot(time,speed_kmh)
axs2[0,0].plot(time,max_speed,color='grey', linestyle='dashed', linewidth=0.5)
axs2[0,0].set_title('Speed (Km/h) vs time (s)')
axs2[0,0].set_xlabel('Time (s)')
axs2[0,0].set_ylabel('Speed (Km/h)')
axs2[0,0].legend(['Speed', 'Max Speed'])
axs_state0 = axs2[0,0].twinx()
# axs_state0.plot(time,state,color='orange', linestyle='dashed', linewidth=1)

axs2[0,1].plot(time,gearbox_heat_kw)
axs2[0,1].plot(time,chain_heat_kw)
axs2[0,1].plot(time,motor_heat_kw)
axs2[0,1].plot(time,inverter_heat_kw)
axs2[0,1].plot(time,bp_heat_kw)
axs2[0,1].legend(['Gearbox', 'Chain', 'Motor', 'Inverter', 'BP'])
axs2[0,1].set_title('Heat production (kW) vs time')
axs2[0,1].set_xlabel('Time (s)')
axs2[0,1].set_ylabel('Heat production (kW)')

axs2[1,0].plot(time,bp_voltage_loaded)
axs2[1,0].set_title('BP Voltage (V) vs time (s)')
axs2[1,0].set_xlabel('Time (s)')
axs2[1,0].set_ylabel('BP Voltage (V)')
axs2[1,0].set_ylim([0.95*min(bp_voltage_loaded),1.05*max(bp_voltage_loaded)])
axs2[1,0].legend(['Voltage'])
# axs2_current0 = axs2[1,0].twinx()
# axs2_current0.plot(time,bp_current,color='orange')
# axs2_current0.set_ylabel('Current (A)')

axs2[1,1].plot(time,bp_temperature)
axs2[1,1].set_title('BP Temperature (max ' + str(round(max(bp_temperature),2)) +') vs time')
axs2[1,1].set_xlabel('Time (s)')
axs2[1,1].set_ylabel('Temperature (C)')

# Maximize all the figures
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# Show the figures
plt.show()