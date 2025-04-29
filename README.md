# RaceSimulation
Python Script for simulating the Performance of a bike for a given circuit

**Running**

To run the script in interactive mode:
python -i lap_time.py



**Required Data:**

Circuit
  - Air pressure
  - Temperature
  - Max speed & distance (in circuit.xlsx)

SOC data
  - Voltage drop
  - Time for discharge
  - Current
  - Voltage (in soc_curve.xlsx)

BP
  - Cells in series
  - Cells in parallel
  - Start voltage
  - End voltage
  - DC resistance
  - Cell total weight
  - Cell heat capacity
  - Cooling Material weight x5
  - Cooling Material heat capacity x5
  - PCM Weight
  - voltagedrop (V/A)

BDU
  - Max current

Motor
  - Max RPM
  - Max Torque
  - Max Power
  - Efficiency

Inverter
  - Efficiency

Transmission
  - First gear ratio
  - Second gear ratio
  - Chain ratio
  - Gearbox efficiency
  - Chain efficiency

Wheels
  - Rear wheel diameter

Resistance
  - Rolling resistance coefficient
  - Frontal area
  - Drag coefficient

Brakes:
  - Braking deceleratiom


Mass:
  - Rider
  - Motorcycle



**Settings**

  - Laps
  - Simulation timestep
