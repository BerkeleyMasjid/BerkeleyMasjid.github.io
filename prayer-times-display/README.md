Setting up raspberry pi

# FullPageOS Image

Download and flash fullpageos

# Wifi

set wifi settings in fullpageos-wpa-supplicant.txt

# Website URL

set url in fullpageos.txt

# Timezone

set time zone by ssh'ing into fullpageos.local
default user: pi
default pass: raspberry

run sudo raspi-config
under Localization, set the timezone

# Screen size

black borders along the edge of the screen can be reduced using the following settings
1. ssh into the raspberry pi with the same credentials as in the timezone section above.
1. type: sudo nano /boot/config.txt
1. set this line: "#disable_overscan=1" to this "disable_overscan=1" (without the quotes)
1. hit ctrl+x then y then Enter
1. type: sudo reboot


# TODO:
Optimize isha iqama transition from 9:30 to 10:15. Ideally should go 9:30 -> 10 -> 10:15 to reduce complaints.

