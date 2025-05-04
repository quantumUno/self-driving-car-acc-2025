import numpy as np
import platform
import os
import json
import time
from quanser.devices.exceptions import DeviceError
from quanser.hardware import HIL, HILError, PWMMode, MAX_STRING_LENGTH, Clock
from quanser.hardware.enumerations import BufferOverflowMode
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout

from pal.utilities.vision import Camera2D, Camera3D
from pal.utilities.lidar import Lidar
from pal.utilities.stream import BasicStream
from pal.utilities.math import Calculus
from pal.products.qcar_config import QCar_check
from os.path import realpath, join, exists, dirname

try:
    IS_PHYSICAL_QCAR = ('nvidia' == os.getlogin()) \
        and ('aarch64' == platform.machine())
except OSError:
    IS_PHYSICAL_QCAR = False

QCAR_CONFIG_PATH = join(dirname(realpath(__file__)), 'qcar_config.json')

if not exists(QCAR_CONFIG_PATH):
    try:
        QCar_check(IS_PHYSICAL_QCAR)
        QCAR_CONFIG = json.load(open(QCAR_CONFIG_PATH, 'r'))
    except HILError as e:
        print('QCar configuration file loading unsuccessful')
        print(e.get_error_message())
else:
    QCAR_CONFIG = json.load(open(QCAR_CONFIG_PATH, 'r'))

class QCar():
    def __init__(
            self,
            readMode=0,
            frequency=500,
            pwmLimit=0.5,  # Increased from 0.3 to 0.5
            steeringBias=0
        ):
        self.card = HIL()
        self.hardware = IS_PHYSICAL_QCAR
        
        print("IS_PHYSICAL_QCAR:", IS_PHYSICAL_QCAR)
        print("Hardware mode:", self.hardware)
        print("QCAR_CONFIG:", QCAR_CONFIG)

        if self.hardware:
            boardIdentifier = "0"
            self.carType = QCAR_CONFIG['cartype']
        else:
            boardIdentifierVirtualQCar = "0@tcpip://localhost:18960?nagle='off'"
            self.carType = 0
            print("Car type (virtual):", self.carType)

        if self.hardware:
            try:
                print("Opening HIL card for physical QCar on boardIdentifier:", boardIdentifier)
                self.card.open(QCAR_CONFIG["carname"], boardIdentifier)
                print("HIL card opened successfully for physical QCar")
                print("Card valid:", self.card.is_valid())
            except HILError as e:
                print("Failed to open HIL card for physical QCar:", e.get_error_message())
                print("Error code:", e.error_code)
                raise
        else:
            try:
                print("Opening HIL card for virtual QCar on boardIdentifier:", boardIdentifierVirtualQCar)
                self.card.open('qcar2', boardIdentifierVirtualQCar)
                print("HIL card opened successfully for virtual QCar")
                print("Card valid:", self.card.is_valid())
            except HILError as e:
                print("Failed to open HIL card for virtual QCar:", e.get_error_message())
                print("Error code:", e.error_code)
                raise

        self.readMode = readMode
        self.io_task_running = False
        self.pwmLimit = pwmLimit
        self.steeringBias = steeringBias
        self.frequency = frequency

        self._configure_parameters()

        if self.card.is_valid():
            self._set_options()

            if self.hardware and self.readMode == 1:
                self._create_io_task()
            print('QCar configured successfully.')
        else:
            print("HIL card is not valid after open attempt")

        self.motorCurrent = np.zeros(1, dtype=np.float64)
        self.batteryVoltage = np.zeros(1, dtype=np.float64)
        self.motorEncoder = np.zeros(1, dtype=np.int32)
        self.motorTach = np.zeros(1, dtype=np.float64)
        self.accelerometer = np.zeros(3, dtype=np.float64)
        self.gyroscope = np.zeros(3, dtype=np.float64)

        self.START_TIME = time.time()

    def _elapsed_time(self):
        return time.time() - self.START_TIME

    def _configure_parameters(self):
        self.ENCODER_COUNTS_PER_REV = 720.0
        self.WHEEL_TRACK = 0.172

        self.WHEEL_RADIUS = QCAR_CONFIG['WHEEL_RADIUS']
        self.WHEEL_BASE = QCAR_CONFIG['WHEEL_BASE']
        self.PIN_TO_SPUR_RATIO = QCAR_CONFIG['PIN_TO_SPUR_RATIO']
        self.CPS_TO_MPS = (1/(self.ENCODER_COUNTS_PER_REV*4)
            * self.PIN_TO_SPUR_RATIO * 2*np.pi * self.WHEEL_RADIUS)

        self.WRITE_PWM_CHANNELS = np.array(
            QCAR_CONFIG['WRITE_PWM_CHANNELS'], 
            dtype=np.int32)

        if not self.hardware:
            self.WRITE_OTHER_CHANNELS = np.array([1000, 11000], dtype=np.int32)  # Steering, throttle only
            self.WRITE_DIGITAL_CHANNELS = np.array([], dtype=np.int32)
            self.READ_ANALOG_CHANNELS = np.array([], dtype=np.int32)
            self.READ_ENCODER_CHANNELS = np.array(
                QCAR_CONFIG['READ_ENCODER_CHANNELS'], 
                dtype=np.uint32)
            self.READ_OTHER_CHANNELS = np.array(
                QCAR_CONFIG['READ_OTHER_CHANNELS'],
                dtype=np.int32
            )

            self.writePWMBuffer = np.zeros(1, dtype=np.float64)
            self.writeDigitalBuffer = np.zeros(0, dtype=np.int8)
            self.writeOtherBuffer = np.zeros(2, dtype=np.float64)  # Steering, throttle only
            self.readAnalogBuffer = np.zeros(0, dtype=np.float64)
            self.readEncoderBuffer = np.zeros(
                len(self.READ_ENCODER_CHANNELS), 
                dtype=np.int32)
            self.readOtherBuffer = np.zeros(
                len(self.READ_OTHER_CHANNELS), 
                dtype=np.float64)
        else:
            self.WRITE_OTHER_CHANNELS = np.array(
                QCAR_CONFIG['WRITE_OTHER_CHANNELS'],
                dtype=np.int32
            )
            self.WRITE_DIGITAL_CHANNELS = np.array(
                QCAR_CONFIG['WRITE_DIGITAL_CHANNELS'],
                dtype=np.int32
            )
            self.READ_ANALOG_CHANNELS = np.array(
                QCAR_CONFIG['READ_ANALOG_CHANNELS'], 
                dtype=np.int32)
            self.READ_ENCODER_CHANNELS = np.array(
                QCAR_CONFIG['READ_ENCODER_CHANNELS'], 
                dtype=np.uint32)
            self.READ_OTHER_CHANNELS = np.array(
                QCAR_CONFIG['READ_OTHER_CHANNELS'],
                dtype=np.int32
            )

            self.writePWMBuffer = np.zeros(
                QCAR_CONFIG['writePWMBuffer'], 
                dtype=np.float64)
            self.writeDigitalBuffer = np.zeros(
                QCAR_CONFIG['writeDigitalBuffer'], 
                dtype=np.int8)
            self.writeOtherBuffer = np.zeros(
                QCAR_CONFIG['writeOtherBuffer'], 
                dtype=np.float64)
            self.readAnalogBuffer = np.zeros(
                QCAR_CONFIG['readAnalogBuffer'], 
                dtype=np.float64)
            self.readEncoderBuffer = np.zeros(
                QCAR_CONFIG['readEncoderBuffer'], 
                dtype=np.int32)
            self.readOtherBuffer = np.zeros(
                QCAR_CONFIG['readOtherBuffer'], 
                dtype=np.float64)

    def _set_options(self):
        if self.carType in [1, 0]:
            self.card.set_pwm_mode(
                np.array([0], dtype=np.uint32),
                1,
                np.array([PWMMode.DUTY_CYCLE], dtype=np.int32)
            )
            self.card.set_pwm_frequency(
                np.array([0], dtype=np.uint32),
                1,
                np.array([60e6/4096], dtype=np.float64)
            )
            if self.hardware:
                self.card.write_digital(
                    np.array([40], dtype=np.uint32),
                    1,
                    np.zeros(1, dtype=np.float64)
                )

        boardOptionsString = ("steer_bias=" + str(self.steeringBias)
            + ";motor_limit=" + str(self.pwmLimit) + ';')
        self.card.set_card_specific_options(
            boardOptionsString,
            MAX_STRING_LENGTH
        )

        self.card.set_encoder_quadrature_mode(
            self.READ_ENCODER_CHANNELS,
            len(self.READ_ENCODER_CHANNELS),
            np.array([4],
            dtype=np.uint32)
        )
        self.card.set_encoder_filter_frequency(
            self.READ_ENCODER_CHANNELS,
            len(self.READ_ENCODER_CHANNELS),
            np.array([60e6/1],
            dtype=np.uint32)
        )
        self.card.set_encoder_counts(
            self.READ_ENCODER_CHANNELS,
            len(self.READ_ENCODER_CHANNELS),
            np.zeros(1, dtype=np.int32)
        )

    def _create_io_task(self):
        try:
            self.readTask = self.card.task_create_reader(
                int(self.frequency),
                self.READ_ANALOG_CHANNELS,
                len(self.READ_ANALOG_CHANNELS),
                self.READ_ENCODER_CHANNELS,
                len(self.READ_ENCODER_CHANNELS),
                None,
                0,
                self.READ_OTHER_CHANNELS,
                len(self.READ_OTHER_CHANNELS)
            )
        except HILError as e:
            print(f"Failed to create reader task: {e.get_error_message()} (Error code: {e.error_code})")
            raise

        if self.readTask is None:
            raise Exception("Task creation returned None")

        print(f"Task created successfully: {self.readTask}")

        if self.hardware:
            self.card.task_set_buffer_overflow_mode(
                self.readTask,
                BufferOverflowMode.OVERWRITE_ON_OVERFLOW
            )
        else:
            self.card.task_set_buffer_overflow_mode(
                self.readTask,
                BufferOverflowMode.WAIT_ON_OVERFLOW
            )

        self.card.task_start(
            self.readTask,
            Clock.HARDWARE_CLOCK_0,
            self.frequency,
            2**32-1
        )
        self.io_task_running = True

    def terminate(self):
        try:
            if self.carType in [1]:
                self.write(0, 0, np.zeros(8, dtype=np.float64))
            else:
                self.write(0, 0, np.zeros(16, dtype=np.int8))

            if self.readMode:
                self.card.task_stop(self.readTask)
            self.card.close()

        except HILError as h:
            print(h.get_error_message())

    def read_write_std(self, throttle, steering, LEDs=None):
        self.write(throttle, steering, LEDs)
        self.read()

    def read(self):
        if not (self.hardware or self.io_task_running) and self.readMode:
            self._create_io_task()

        try:
            self.currentTimeStamp = self._elapsed_time()
            if self.readMode == 1:
                self.card.task_read(
                    self.readTask,
                    1,
                    self.readAnalogBuffer,
                    self.readEncoderBuffer,
                    None,
                    self.readOtherBuffer
                )
            else:
                self.card.read(
                    self.READ_ANALOG_CHANNELS,
                    len(self.READ_ANALOG_CHANNELS),
                    self.READ_ENCODER_CHANNELS,
                    len(self.READ_ENCODER_CHANNELS),
                    None,
                    0,
                    self.READ_OTHER_CHANNELS,
                    len(self.READ_OTHER_CHANNELS),
                    self.readAnalogBuffer,
                    self.readEncoderBuffer,
                    None,
                    self.readOtherBuffer
                )
        except HILError as h:
            print(h.get_error_message())
        finally:
            self.motorCurrent = np.zeros(1, dtype=np.float64) if len(self.readAnalogBuffer) < 1 else self.readAnalogBuffer[0]
            self.batteryVoltage = np.zeros(1, dtype=np.float64) if len(self.readAnalogBuffer) < 2 else self.readAnalogBuffer[1]
            self.gyroscope = self.readOtherBuffer[0:3]
            self.accelerometer = self.readOtherBuffer[3:6]
            self.motorEncoder = self.readEncoderBuffer
            # Debug raw tachometer reading
            raw_tach = self.readOtherBuffer[-1] if len(self.readOtherBuffer) > 0 else 0
            # print(f"Raw tachometer reading: {raw_tach:.2f} counts/s, CPS_TO_MPS: {self.CPS_TO_MPS:.6f}")
            self.motorTach = raw_tach * self.CPS_TO_MPS

    def write(self, throttle, steering, LEDs=None):
        if not (self.hardware or self.io_task_running) and self.readMode:
            self._create_io_task()

        if self.carType in [1]:
            self.writeOtherBuffer[0] = -np.clip(steering, -0.6, 0.6)
            self.writePWMBuffer = -np.clip(throttle, -self.pwmLimit, self.pwmLimit)
            if LEDs is not None and self.hardware:
                self.writeOtherBuffer[1:9] = LEDs
        else:  # carType 0 (virtual) or 2
            self.writeOtherBuffer[0] = np.clip(steering, -0.6, 0.6)
            self.writeOtherBuffer[1] = np.clip(throttle, -self.pwmLimit, self.pwmLimit)
            if LEDs is not None and self.hardware:  # Skip LEDs for virtual QCar
                self.writeDigitalBuffer[0:4] = LEDs[0:4]
                self.writeDigitalBuffer[4] = LEDs[4]
                self.writeDigitalBuffer[5] = LEDs[4]
                self.writeDigitalBuffer[6] = LEDs[4]
                self.writeDigitalBuffer[7] = LEDs[4]
                self.writeDigitalBuffer[8] = LEDs[5]
                self.writeDigitalBuffer[9] = LEDs[5]
                self.writeDigitalBuffer[10] = LEDs[6]
                self.writeDigitalBuffer[11] = LEDs[6]
                self.writeDigitalBuffer[12] = LEDs[6]
                self.writeDigitalBuffer[13] = LEDs[7]
                self.writeDigitalBuffer[14] = LEDs[7]
                self.writeDigitalBuffer[15] = LEDs[7]

        try:
            if self.carType in [1]:
                self.card.write(
                    None,
                    0,
                    self.WRITE_PWM_CHANNELS,
                    len(self.WRITE_PWM_CHANNELS),
                    None,
                    0,
                    self.WRITE_OTHER_CHANNELS,
                    len(self.WRITE_OTHER_CHANNELS),
                    None,
                    self.writePWMBuffer,
                    None,
                    self.writeOtherBuffer
                )
            else:
                self.card.write(
                    None,
                    0,
                    None,
                    0,
                    self.WRITE_DIGITAL_CHANNELS,
                    len(self.WRITE_DIGITAL_CHANNELS),
                    self.WRITE_OTHER_CHANNELS,
                    len(self.WRITE_OTHER_CHANNELS),
                    None,
                    None,
                    self.writeDigitalBuffer,
                    self.writeOtherBuffer
                )
        except HILError as h:
            print(h.get_error_message())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.terminate()

class QCarCameras:
    def __init__(
            self,
            frameWidth=820,
            frameHeight=410,
            frameRate=30,
            enableRight=False,
            enableBack=False,
            enableLeft=False,
            enableFront=False,
        ):
        if QCAR_CONFIG['cartype'] in [0, 1]:
            enable = [enableRight, enableBack, enableLeft, enableFront]
        elif QCAR_CONFIG['cartype'] == 2:
            enable = [enableRight, enableBack, enableFront, enableLeft]
        self.csi = []
        for i in range(4):
            if enable[i]:
                if IS_PHYSICAL_QCAR:
                    cameraId = str(i)
                else:
                    cameraId = str(i) + "@tcpip://localhost:" + str(18961 + i)

                self.csi.append(
                    Camera2D(
                        cameraId=cameraId,
                        frameWidth=frameWidth,
                        frameHeight=frameHeight,
                        frameRate=frameRate
                    )
                )
            else:
                self.csi.append(None)

        self.csiRight = self.csi[QCAR_CONFIG['csiRight']]
        self.csiBack = self.csi[QCAR_CONFIG['csiBack']]
        self.csiLeft = self.csi[QCAR_CONFIG['csiLeft']]
        self.csiFront = self.csi[QCAR_CONFIG['csiFront']]

    def readAll(self):
        flags = []
        for c in self.csi:
            if c is not None:
                flags.append(c.read())
        return flags

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for c in self.csi:
            if c is not None:
                c.terminate()

    def terminate(self):
        for c in self.csi:
            if c is not None:
                c.terminate()

class QCarLidar(Lidar):
    def __init__(
            self,
            numMeasurements=384,
            rangingDistanceMode=2,
            interpolationMode=0,
            interpolationMaxDistance=0,
            interpolationMaxAngle=0,
            enableFiltering=True,
            angularResolution=1*np.pi/180
        ):
        try:
            if IS_PHYSICAL_QCAR:
                self.url = QCAR_CONFIG['lidarurl']
            else:
                self.url = "tcpip://localhost:18966"

            super().__init__(
                type='RPLidar',
                numMeasurements=numMeasurements,
                rangingDistanceMode=rangingDistanceMode,
                interpolationMode=interpolationMode,
                interpolationMaxDistance=interpolationMaxDistance,
                interpolationMaxAngle=interpolationMaxAngle
            )

        except DeviceError as e:
            print("Lidar Error")
            print(e.get_error_message())

        self.enableFiltering = enableFiltering
        self.angularResolution = angularResolution
        self._phi = np.linspace(
            0,
            2*np.pi,
            np.int_(np.round(2*np.pi/self.angularResolution))
            )

    def read(self):
        new = super().read()
        if self.enableFiltering:
            self.angles, self.distances = self.filter_rplidar_data(
                self.angles, self.distances
            )
        return new

    def filter_rplidar_data(self, angles, distances):
        phiRes = self.angularResolution
        ids = (distances == 0).squeeze()
        rMeas = np.delete(distances, ids)
        phiMeas = np.delete(angles, ids)
        if phiMeas.size == 0:
            return phiMeas, rMeas

        phiMeas, ids = np.unique(phiMeas, return_index=True)
        rMeas = rMeas[ids]

        rFiltered = np.interp(
            self._phi,
            phiMeas,
            rMeas,
            period=2*np.pi
        )

        ids = np.diff(phiMeas) > 1.1 * phiRes
        ids_lb = np.append(ids, False)
        ids_ub = np.append(False, ids)

        lb = np.int_(np.ceil(phiMeas[ids_lb]/phiRes))
        ub = np.int_(np.floor(phiMeas[ids_ub]/phiRes))
        for i in range(lb.size):
            rFiltered[lb[i]:ub[i]] = 0

        phiMeasMin = np.int_(np.round(phiMeas[0]/phiRes))
        phiMeasMax = np.int_(np.round(phiMeas[-1]/phiRes))
        rFiltered[0:phiMeasMin] = 0
        rFiltered[phiMeasMax+1:] = 0

        return self._phi.astype('float32'), rFiltered.astype('float32')

class QCarRealSense(Camera3D):
    def __init__(
            self,
            mode='RGB&DEPTH',
            frameWidthRGB=1920,
            frameHeightRGB=1080,
            frameRateRGB=30,
            frameWidthDepth=1280,
            frameHeightDepth=720,
            frameRateDepth=15,
            frameWidthIR=1280,
            frameHeightIR=720,
            frameRateIR=15,
            readMode=1,
            focalLengthRGB=np.array([[None], [None]], dtype=np.float64),
            principlePointRGB=np.array([[None], [None]], dtype=np.float64),
            skewRGB=None,
            positionRGB=np.array([[None], [None], [None]], dtype=np.float64),
            orientationRGB=np.array(
                [[None, None, None], [None, None, None], [None, None, None]],
                dtype=np.float64),
            focalLengthDepth=np.array([[None], [None]], dtype=np.float64),
            principlePointDepth=np.array([[None], [None]], dtype=np.float64),
            skewDepth=None,
            positionDepth=np.array([[None], [None], [None]], dtype=np.float64),
            orientationDepth=np.array(
                [[None, None, None], [None, None, None], [None, None, None]],
                dtype=np.float64)
        ):

        if IS_PHYSICAL_QCAR:
            deviceId = '0'
        else:
            deviceId = "0@tcpip://localhost:18965"
            frameWidthRGB = 640
            frameHeightRGB = 480
            frameRateRGB = 30
            frameWidthDepth = 640
            frameHeightDepth = 480
            frameRateDepth = 15
            frameWidthIR = 640
            frameHeightIR = 480
            frameRateIR = 30

        super().__init__(
            mode,
            frameWidthRGB,
            frameHeightRGB,
            frameRateRGB,
            frameWidthDepth,
            frameHeightDepth,
            frameRateDepth,
            frameWidthIR,
            frameHeightIR,
            frameRateIR,
            deviceId,
            readMode,
            focalLengthRGB,
            principlePointRGB,
            skewRGB,
            positionRGB,
            orientationRGB,
            focalLengthDepth,
            principlePointDepth,
            skewDepth,
            positionDepth,
            orientationDepth
        )

class QCarGPS:
    def __init__(self, initialPose=[0, 0, 0], calibrate=False):
        self._need_calibrate = calibrate
        if IS_PHYSICAL_QCAR:
            self.__initLidarToGPS(initialPose)

        self._timeout = Timeout(seconds=0, nanoseconds=1)

        self.position = np.zeros((3))
        self.orientation = np.zeros((3))

        self._gps_data = np.zeros((6), dtype=np.float32)
        self._gps_client = BasicStream(
            uri="tcpip://localhost:18967",
            agent='C',
            receiveBuffer=np.zeros(6, dtype=np.float32),
            sendBufferSize=1,
            recvBufferSize=(self._gps_data.size * self._gps_data.itemsize),
            nonBlocking=True
        )
        t0 = time.time()
        while not self._gps_client.connected:
            if time.time() - t0 > 5:
                print("Couldn't Connect to GPS Server")
                return
            self._gps_client.checkConnection()

        self.scanTime = 0
        self.angles = np.zeros(384)
        self.distances = np.zeros(384)

        self._lidar_data = np.zeros(384*2 + 1, dtype=np.float64)
        self._lidar_client = BasicStream(
            uri="tcpip://localhost:18968",
            agent='C',
            receiveBuffer=np.zeros(384*2 + 1, dtype=np.float64),
            sendBufferSize=1,
            recvBufferSize=8*(384*2 + 1),
            nonBlocking=True
        )
        t0 = time.time()
        while not self._lidar_client.connected:
            if time.time() - t0 > 5:
                print("Couldn't Connect to Lidar Server")
                return
            self._lidar_client.checkConnection()

        self.enableFiltering = True
        self.angularResolution = 1*np.pi/180
        self._phi = np.linspace(
            0,
            2*np.pi,
            np.int_(np.round(2*np.pi/self.angularResolution))
        )

    def __initLidarToGPS(self, initialPose):
        self.__initialPose = initialPose

        self.__stopLidarToGPS()

        if self._need_calibrate:
            self.__calibrate()
            time.sleep(16)
        if os.path.exists(os.path.join(os.getcwd(), 'angles_new.mat')):
            self.__emulateGPS()
            time.sleep(4)
            print('GPS Server started.')
        else:
            print('Calibration files not found, please set the argument \'calibration\' to True.')
            exit(1)

    def __stopLidarToGPS(self):
        os.system(
            'quarc_run -t tcpip://localhost:17000 -q -Q '
            + QCAR_CONFIG['lidarToGps']
        )

    def __calibrate(self):
        print('Calibrating QCar at position ', self.__initialPose[0:2],
            ' (m) and heading ', self.__initialPose[2], ' (rad).')

        captureScanfile = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            '../../../resources/applications/QCarScanMatching/'
                + QCAR_CONFIG['captureScan']
        ))

        os.system(
            'quarc_run -t tcpip://localhost:17000 '
            + captureScanfile + ' -d ' + os.getcwd()
        )

    def __emulateGPS(self):
        lidarToGPSfile = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            '../../../resources/applications/QCarScanMatching/'
                + QCAR_CONFIG['lidarToGps']
        ))
        os.system(
            'quarc_run -r -t tcpip://localhost:17000 '
            + lidarToGPSfile + ' -d ' + os.getcwd()
            + ' -pose_0 ' + str(self.__initialPose[0])
            + ',' + str(self.__initialPose[1])
            + ',' + str(self.__initialPose[2])
        )

    def readGPS(self):
        recvFlag, bytesReceived = self._gps_client.receive(
            iterations=1,
            timeout=self._timeout)

        if recvFlag:
            self.position = self._gps_client.receiveBuffer[0:3]
            self.orientation = self._gps_client.receiveBuffer[3:6]

        return recvFlag

    def readLidar(self):
        recvFlag, bytesReceived = self._lidar_client.receive(
            iterations=1,
            timeout=self._timeout)

        if recvFlag:
            self.scanTime = self._lidar_client.receiveBuffer[0]
            self.distances = self._lidar_client.receiveBuffer[1:385]
            self.angles = self._lidar_client.receiveBuffer[385:769]

            self.angles, self.distances = self.filter_rplidar_data(
                self.angles,
                self.distances
            )

        return recvFlag

    def filter_rplidar_data(self, angles, distances):
        phiRes = self.angularResolution
        ids = (distances == 0)
        phiMeas = np.delete(angles, ids)
        rMeas = np.delete(distances, ids)
        if phiMeas.size == 0:
            return phiMeas, rMeas

        phiMeas, ids = np.unique(phiMeas, return_index=True)
        rMeas = rMeas[ids]

        rFiltered = np.interp(
            self._phi,
            phiMeas,
            rMeas,
            period=2*np.pi
        )

        ids = np.diff(phiMeas) > 1.1 * phiRes
        ids_lb = np.append(ids, False)
        ids_ub = np.append(False, ids)

        lb = np.int_(np.ceil(phiMeas[ids_lb]/phiRes))
        ub = np.int_(np.floor(phiMeas[ids_ub]/phiRes))
        for i in range(lb.size):
            rFiltered[lb[i]:ub[i]] = 0

        phiMeasMin = np.int_(np.round(phiMeas[0]/phiRes))
        phiMeasMax = np.int_(np.round(phiMeas[-1]/phiRes))
        rFiltered[0:phiMeasMin] = 0
        rFiltered[phiMeasMax+1:] = 0

        return self._phi, rFiltered

    def terminate(self):
        self._gps_client.terminate()
        self._lidar_client.terminate()
        if IS_PHYSICAL_QCAR:
            self.__stopLidarToGPS()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.terminate()
