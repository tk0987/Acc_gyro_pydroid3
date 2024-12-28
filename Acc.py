import kivy
from plyer import accelerometer, gyroscope
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from time import sleep
from datetime import datetime

# Covariance, means, variances, and process noise
R_accel = np.array([[2.82491373e-01, -2.11905161e-01, 1.38368222e-03],
                    [-2.11905161e-01, 1.82725701e-01, -1.34578881e-03],
                    [1.38368222e-03, -1.34578881e-03, 1.59464906e-04]]) * 10.0

R_gyro = np.array([[4.51203699e-03, -1.26373919e-03, 9.84813925e-04],
                   [-1.26373919e-03, 1.54366492e-03, -3.43728683e-04],
                   [9.84813925e-04, -3.43728683e-04, 6.33979341e-04]]) * 10

means_accel = np.array([0.0, 0.0, 0.0])
means_gyro = np.array([0.0, 0.0, 0.0])

variances_accel = np.array([0.28247228528630045, 0.1827133549105913, 1.5945413107437287e-03]) * 5
variances_gyro = np.array([0.04511732118901754, 0.01543560623238229, 6.339365047677065e-03]) * 5

Q_accel = np.array([[14.03147346, -3.44430711, -1.59433783],
                    [-3.44430711, 9.76366978, 0.65427711],
                    [-1.59433783, 0.65427711, 5.20911763]]) * 5

Q_gyro = np.array([[878.7735367, -70.37832705, 42.61623472],
                   [-70.37832705, 1133.27086257, 221.97269171],
                   [42.61623472, 221.97269171, 2202.26991191]]) * 8

F = np.eye(6)  # Example state transition model
B = np.zeros((6, 3))  # Example control input model
H = np.eye(6)  # Example observation model
x0 = np.concatenate((means_accel, means_gyro))  # Initial state estimate
P0 = np.diag(np.concatenate((variances_accel, variances_gyro)))  # Initial covariance estimate
Q = np.block([
    [Q_accel, np.zeros((3, 3))],
    [np.zeros((3, 3)), Q_gyro]
])  # Combined process noise covariance
R = np.block([
    [R_accel, np.zeros((3, 3))],
    [np.zeros((3, 3)), R_gyro]
])  # Combined measurement noise covariance


class KalmanFilter:
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        return self.x


class SensorApp:
    def __init__(self):
        accelerometer.enable()
        gyroscope.enable()

        self.kf = KalmanFilter(F=F, B=B, H=H, Q=Q, R=R, x0=x0, P0=P0)

        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.f=open('/storage/emulated/0/skanfon/res.txt','w')

        accel_data = accelerometer.acceleration
        while accel_data is None:
            accel_data = accelerometer.acceleration
        if accel_data:
            ax, ay, az = accel_data
            norm = np.sqrt(ax * ax + ay * ay + az * az)
            self.alpha = np.arccos(ax / norm) * 180 / np.pi
            self.beta = np.arccos(ay / norm) * 180 / np.pi
            self.gamma = np.arccos(az / norm) * 180 / np.pi

        self.ang = np.zeros((10, 3))
        self.a = 0
        self.b = 0
        self.c = 0
        self.f.write(f"{self.alpha}\t{self.beta}\t{self.gamma}\n")

    def update_sensors(self):
        try:
            accel_data = accelerometer.acceleration
            gyro_data = gyroscope.rotation
            current_time = datetime.now()
            ftime = current_time.strftime("%Y_%m_%d-%H_%M_%S_%f")[:-3]

            if accel_data and gyro_data:
                ax, ay, az = accel_data
                gx, gy, gz = gyro_data

                gx, gy, gz = gx * 180 / np.pi, gy * 180 / np.pi, gz * 180 / np.pi

                u = np.array([gx, gy, gz])

                z = np.concatenate(([ax, ay, az], [gx, gy, gz]))

                pred = self.kf.predict(u)
                upd = self.kf.update(z)
                upd = np.round(upd, 3)
                norma = np.linalg.norm(upd[:3])

                accel_alpha = np.round(np.arccos(upd[0] / norma) * 180 / np.pi, 2)
                accel_beta = np.round(np.arccos(upd[1] / norma) * 180 / np.pi, 2)
                accel_gamma = np.round(np.arccos(upd[2] / norma) * 180 / np.pi, 2)

                self.alpha, self.beta, self.gamma = accel_alpha, accel_beta, accel_gamma

                self.update_ang(upd[3], upd[4], upd[5])
#                print(f"Accelerometer: x={upd[0]}, y={upd[1]}, z={upd[2]}")
#                print(f"Gyroscope: x={upd[3]}, y={upd[4]}, z={upd[5]}")
#                print(f"Orientation: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")
                self.f.write(f"{ftime}\t{self.a}\t{self.b}\t{self.c}\n")
            else:
                print("Accelerometer or Gyroscope data unavailable")
        except Exception as e:
            print(f"Error occurred: {e}")

    def update_ang(self, anx, any, anz):
        def append_new_row(arr, new_row):
            arr = np.roll(arr, shift=-1, axis=0)
            arr[-1] = new_row
            return arr

        self.ang = append_new_row(self.ang, np.asarray([anx / 60, any / 60, anz / 60]))
        self.a += np.mean(self.ang[:, 0])
        self.b += np.mean(self.ang[:, 1])
        self.c += np.mean(self.ang[:, 2])

    def run(self):
        try:
            while True:
                self.update_sensors()
                sleep(1 / 60)  # Update at 60 Hz
        finally:
            accelerometer.disable()
            gyroscope.disable()
            self.f.close()


if __name__ == "__main__":
    app = SensorApp()
    app.run()
