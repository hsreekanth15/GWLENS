#Code to generate signal for TDFstat.
# Original code written by Anirudh Nemmani (CAMK PAN) for TDFstat-PE
#Adapted to GWLENS by Sreekanth Harikumar (CAMK PAN) and modified to be compatible with the GWLENS framework.

from . import constants
import numpy as np


class TimeSeries:
    """
    A TimeSeries object represents a time-domain signal generated for a given detector and source parameters. It encapsulates the necessary computations to generate the signal based on the detector's properties and the source's characteristics.

    Parameters
    ----------
    detector : object
        An object representing the detector, which contains properties such as segment, band, overlap, start_time, delta_t, duration, starting_frequency, sampling_frequency, bandwidth, time_array, time_square_array, t_plus_t2_over_2, and time_domain_strain.
    
    Attributes
    ----------
    detector : object
        An object representing the detector.
    segment : float
        The segment of the signal.
    band : float
        The band of the signal.
    overlap : float
        The overlap of the signal.
    start_time : float
        The start time of the signal.
    delta_t : float
        The time step of the signal.
    duration : float
        The duration of the signal.
    starting_frequency : float
        The starting frequency of the signal.
    sampling_frequency : float
        The sampling frequency of the signal.
    bandwidth : float
        The bandwidth of the signal.
    time_array : numpy.ndarray
        The time array of the signal.
    time_square_array : numpy.ndarray
        The square of the time array of the signal.
    t_plus_t2_over_2 : numpy.ndarray
        An array used in computations.
    signal : numpy.ndarray
        The generated signal.
    phase : numpy.ndarray
        The phase of the signal.
    amplitude_mod_a : numpy.ndarray
        The amplitude modulation for the a component.
    amplitude_mod_b : numpy.ndarray
        The amplitude modulation for the b component.
    amplification : numpy.ndarray
        The amplification factor for the signal.
    
    Methods
    -------
    generate_time_domain_signal(**params)
        Generates the time-domain signal based on the provided source parameters.   

    """
    def __init__(self, detector):
        self.detector = detector

        self.segment = detector.segment
        self.band = detector.band
        self.overlap = detector.overlap

        self.start_time = detector.start_time
        self.delta_t = detector.delta_t
        self.duration = detector.duration

        self.starting_frequency = detector.starting_frequency
        self.sampling_frequency = detector.sampling_frequency
        self.bandwidth = detector.bandwidth

        self.time_array = detector.time_array
        self.time_square_array = detector.time_square_array
        self.t_plus_t2_over_2 = detector.t_plus_t2_over_2

        self.signal = None
        self.phase = None
        self.amplitude_mod_a = None
        self.amplitude_mod_b = None
        self.amplification = None

    @staticmethod
    def _generate_source_vector(ra, dec):
        """
        Generate the source vector based on right ascension and declination.    

        Parameters
        ----------
        ra : float
            Right ascension of the source in radians.
        dec : float
            Declination of the source in radians.   
        
        Returns
        -------
        numpy.ndarray
            A 3-element array representing the source vector in Cartesian coordinates.
        """
        cos_dec = np.cos(dec)
        return np.array([
            np.cos(ra) * cos_dec,
            np.sin(ra) * cos_dec,
            np.sin(dec),
        ])

    def _generate_amplitude_modulation(self, ra, dec):
        """
        Generate the amplitude modulation for the signal based on the source parameters.
        From JKS paper: http://arxiv.org/abs/gr-qc/9804014
        
        Parameters
        ----------
        ra : float
            Right ascension of the source in radians.
        dec : float
            Declination of the source in radians.

        Returns
        -------
        tuple
            A tuple containing the amplitude modulation factors for the a and b components.
        """

        ra_mod = (
            ra
            - self.detector.ephemeris.phir
            - constants.C_OMEGA_R * self.time_array
        )

        sin_2psi = np.sin(2.0 * self.detector.orientation_radians)
        cos_2psi = np.cos(2.0 * self.detector.orientation_radians)

        lon = self.detector.longitude_radians
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)
        sin_2lon = np.sin(2.0 * lon)
        cos_2lon = np.cos(2.0 * lon)

        sin_dec = np.sin(dec)
        cos_dec = np.cos(dec)
        sin_2dec = np.sin(2.0 * dec)
        cos_2dec = np.cos(2.0 * dec)

        cos_ra = np.cos(ra_mod)
        sin_ra = np.sin(ra_mod)
        cos_2ra = np.cos(2.0 * ra_mod)
        sin_2ra = np.sin(2.0 * ra_mod)

        a = (
            0.0625 * sin_2psi * (3.0 - cos_2lon) * (3.0 - cos_2dec) * cos_2ra
            - 0.25 * cos_2psi * sin_lon * (3.0 - cos_2dec) * sin_2ra
            + 0.25 * sin_2psi * sin_2lon * sin_2dec * cos_ra
            - 0.5 * cos_2psi * cos_lon * sin_2dec * sin_ra
            + 0.75 * sin_2psi * cos_lon ** 2 * cos_dec ** 2
        )

        b = (
            cos_2psi * sin_lon * sin_dec * cos_2ra
            + 0.25 * sin_2psi * (3.0 - cos_2lon) * sin_dec * sin_2ra
            + cos_2psi * cos_lon * cos_dec * cos_ra
            + 0.25 * sin_2psi * sin_2lon * cos_dec * sin_ra
        )
        return a, b

    @staticmethod
    def _generate_four_amplitudes(
        wobble_angle, inclination, polarization, reference_phase):

        """
        Generate the four amplitudes based on the source parameters.
        From JKS paper: http://arxiv.org/abs/gr-qc/9804014  
        
        Parameters
        ----------
        wobble_angle : float
            The wobble angle of the source in radians.
        inclination : float
            The inclination of the source in radians.
        polarization : float
            The polarization angle of the source in radians.
        reference_phase : float
            The reference phase of the source in radians.
        Returns
        -------
        tuple
            A tuple containing the four amplitudes (a21, a22, a23, a24) based on the source parameters.     
        
        """
        
        sin_w2 = np.sin(wobble_angle) ** 2
        cos_i = np.cos(inclination)
        inc_term = 0.5 * (1.0 + cos_i ** 2)

        cos_2psi = np.cos(2.0 * polarization)
        sin_2psi = np.sin(2.0 * polarization)
        cos_2phi = np.cos(2.0 * reference_phase)
        sin_2phi = np.sin(2.0 * reference_phase)

        a21 = sin_w2 * (inc_term * cos_2psi * cos_2phi - cos_i * sin_2psi * sin_2phi)
        a22 = sin_w2 * (inc_term * sin_2psi * cos_2phi + cos_i * cos_2psi * sin_2phi)
        a23 = -sin_w2 * (inc_term * cos_2psi * sin_2phi + cos_i * sin_2psi * cos_2phi)
        a24 = -sin_w2 * (inc_term * sin_2psi * sin_2phi - cos_i * cos_2psi * cos_2phi)

        return a21, a22, a23, a24

    def _generate_signal_phase(self, frequency, spin_down, ra, dec):
        """
        Generate the signal phase based on the source parameters.

        Parameters
        ----------      
        frequency : float
            The frequency of the signal.
        spin_down : float
            The spin-down rate of the signal.
        ra : float
            Right ascension of the source in radians.
        dec : float
            Declination of the source in radians.   
        """
        nsource = self._generate_source_vector(ra, dec)
        r_dot_n = np.dot(self.detector.ephemeris.DetSSB, nsource)


        # original
        zero_order = (1.0 + r_dot_n) * frequency * self.time_array
        # fixed: delay added to the time; Doppler term uses the full frequency (fpo/2 + frequency)
        #zero_order = frequency * self.time_array + (frequency + 0.5 * self.starting_frequency) * r_dot_n
        first_order = self.t_plus_t2_over_2 * spin_down
        return 2.0 * np.pi * (zero_order + first_order)

    def generate_time_domain_signal(self, **params):

        """
        This method generates the time-domain signal based on the provided parameters.  

        Parameters
        ----------
        params : dict
            A dictionary containing the parameters required for signal generation. Expected keys include:
            - "frequency": The frequency of the signal.
            - "spin_down": The spin-down rate of the signal.
            - "ra": Right ascension of the source.
            - "dec": Declination of the source.
    
        Returns
        -------
        signal : numpy.ndarray
            The generated time-domain signal.
        time_array : numpy.ndarray
            The time array corresponding to the signal.
        """ 
        phase = self._generate_signal_phase(
            params["frequency"],
            params["spin_down"],
            params["ra"],
            params["dec"],
        )

        a, b = self._generate_amplitude_modulation(
            params["ra"], params["dec"]
        )

        a21, a22, a23, a24 = self._generate_four_amplitudes(
            params["wobble_angle"],
            params["inclination"],
            params["polarization"],
            params["reference_phase"],
        )

        #Lensing inclusion (doesn't existn in the TDFstat-PE version)

        amplification = params.get("amplification")
        signal_phase = 2.0 * phase
        envelope = 1.0

        if amplification is not None:
            amplification = np.asarray(amplification)
            if amplification.size != len(self.time_array):
                raise ValueError(
                    "Amplification array length must match time array length"
                )
            signal_phase = 2.0 * phase + np.angle(amplification)
            envelope = np.abs(amplification)

        #Phase of the below equation has to be checked. It is not clear if The current implementation adds it, but this may need to be verified with the physics of the problem.
        signal = envelope*(
            (a * a21 + b * a22) * np.cos(signal_phase)
            + (a * a23 + b * a24) * np.sin(signal_phase)
        )


        if self.detector.time_domain_strain is None:
            raise RuntimeError(
                f"Time-domain strain data not available in detector {self.detector.name}"
            )

        if len(signal) != len(self.detector.time_domain_strain):
            raise ValueError("Signal length must match time-domain strain data length")

        self.phase = phase
        self.amplitude_mod_a = a
        self.amplitude_mod_b = b
        self.amplification = amplification
        self.signal = signal

        self.detector.time_domain_strain += self.signal
        return self.signal, self.time_array