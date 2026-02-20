# unlock.py -> code for the manual unlock function (admin feature) == main code
#-----------------------------------------------------------------------------


# IMPORTS
import serial
import time

print("Trying to connect to Arduino...")

try:
    # Try ACM0 first
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    print("Connected on /dev/ttyACM0")
except:
    
    try:
        
        # If that fails, try USB0
        ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        print("Connected on /dev/ttyUSB0")
     # Handle exception   
    except Exception as e:
        print(f"CRITICAL ERROR: Could not find Arduino! \n{e}")
        exit()


#ACTION
time.sleep(2) # Wait for Arduino to reset
print("Sending Open Command...")
ser.write(b'O')
print("Sent! Did the lock move?")
ser.close()
