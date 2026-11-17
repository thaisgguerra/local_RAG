# Hardware: Parcel Locker

## Main architecture
Each locker bank is controlled by a single-board computer (ARM, 4 GB RAM) with a 10-inch touchscreen. A separate microcontroller board drives the compartment locks and reads the door sensors.
The two boards talk over a serial link at 115200 baud.

## Locks and sensors
Every compartment has an electromagnetic lock and a magnetic door sensor.
The small compartment measures 20 x 30 x 40 cm, the medium 30 x 40 x 50 cm and the large 45 x 50 x 60 cm.
The compartments have no weight sensor. [TO VERIFY] A light curtain for detecting parcels left inside is planned for the next hardware revision.

## Power and connectivity
The locker runs on 110-240 V mains power with a battery backup that keeps the locks working for 4 hours.
Connectivity is Ethernet by default, with an optional 4G modem.

## Production and cost
The locker is assembled by two contract manufacturers. The average production cost is USD 1,900 per 24-compartment unit. [TO VERIFY] The cost may drop about 10% when volume passes 500 units per year.

## Maintenance and lifespan
The estimated lifespan is 7 to 10 years for the structure and 3 to 5 years for the touchscreen.
Preventive maintenance is done every 6 months: lock test, sensor test, screen cleaning and battery check.
