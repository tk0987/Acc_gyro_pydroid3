# Acc_gyro_pydroid3

goal: log gyro/acc data, take photos via camera app and recover 3d object.

its done via backprojection method. but not like in CT - here we project images onto the plane (xy plane) thru volume. and it works.

i dont have root on my phone, so i need to use this "toolchain" to create 3d models. pyjnius dont work for me within pydroid3 and kivy dont like termux...

this method is intended to use with stationary photogrammetric 3d scanners, without the need of integrating positions and orientations. but i provided code for unrooted smartphones - just like a hint.

# 1. run gyro logging script. wait 10 seconds and put it into background

# 2. run camera app and take photos of an object calmly. preserve the radius. 

# 3. copy and rename photos via script

# 4. backprojection now. set resolution. its cpu expensive at least for today
