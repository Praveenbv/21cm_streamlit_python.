import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from numpy.fft import fft, fftshift
from datetime import datetime, timedelta
import csv
import os
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Galactic
from astropy.time import Time
import astropy.units as u

# --- Constants ---
R0 = 8.5            # kpc (Distance to Galactic Center)
V0 = 220            # km/s (Solar Orbital Velocity)
F0 = 1420.405751    # MHz (Hydrogen Rest Frequency)
C = 299792.458      # km/s (Speed of Light)

def capture_rtl_data(center_freq, num_samples, chunk_size=1024000):
    sdr = RtlSdr()
    sdr.sample_rate = 2.4e6
    sdr.center_freq = center_freq
    sdr.gain = 'auto'
    samples = np.array([], dtype=np.complex64)
    try:
        for _ in range(num_samples // chunk_size):
            samples = np.concatenate((samples, sdr.read_samples(chunk_size)))
    finally:
        sdr.close()
    return samples

def return_averaged_spectras(ipdata, chNo, nAvgerages, nSets, npt):
    ipdata = np.asarray(ipdata)
    row = len(ipdata)
    totalSpectrasAsked = nAvgerages * nSets
    NoOfAvailableSpectras = (row // npt) - 1

    if totalSpectrasAsked <= NoOfAvailableSpectras:
        aspecA = np.zeros((npt, nSets))
        for set_idx in range(nSets):
            for I in range(nAvgerages):
                startNo = (I * npt) + (set_idx * nAvgerages * npt) + 1
                endNo = startNo + npt
                if endNo > len(ipdata): 
                    break
                segment = ipdata[startNo:endNo] if ipdata.ndim == 1 else ipdata[startNo:endNo, chNo]
                spc = np.abs(fft(segment)) ** 2
                aspecA[:, set_idx] += spc[:npt]
            aspecA[:, set_idx] = fftshift(aspecA[:, set_idx] / nAvgerages)
        return aspecA
    else:
        st.error("Error: Not enough data for averaging.")
        return np.zeros((2, 1))

def process_data(aa, bb, colNo=1, nfft=128):
    navg = len(aa) // nfft - 3
    nsets = 1
    avgps = return_averaged_spectras(aa, colNo, navg, nsets, nfft)
    avgps2 = return_averaged_spectras(bb, colNo, navg, nsets, nfft)
    return avgps, avgps2

# --- MODIFIED: Tangent Point Method Calculation ---
def calculate_velocity_and_distance(obs_freq_mhz, L):
    # Convert L to radians
    L_rad = np.radians(L)
    
    # 1. Radial Velocity (Vr) from Doppler Shift
    # Astronomical convention: Positive = Moving Away (Redshift)
    # Formula: Vr = c * (f_rest - f_obs) / f_rest
    Vr = C * (F0 - obs_freq_mhz) / F0
    
    # 2. Tangent Point Distance (R)
    # Geometry: R = R0 * sin(L)
    # Only valid for inner galaxy (0 < L < 90)
    R = R0 * np.sin(L_rad)
    
    # 3. Galactic Rotation Velocity (V_rot)
    # Formula: V_rot = Vr + V0 * sin(L)
    # (Assuming Vr is velocity w.r.t LSR/Sun roughly)
    V_rot = Vr + (V0 * np.sin(L_rad))
    
    return {
        'Vr': round(Vr, 2),        # Observed Radial Velocity
        'R': round(abs(R), 2),     # Distance from Galactic Center (Tangent Point)
        'V_rot': round(V_rot, 2)   # Orbital Velocity of the gas
    }

# --- MODIFIED: Plots with Tangent Point Labels ---
def create_plots(avgps, avgps2, num, L, nfft=128, alt=None, az=None, B=None, RA=None, DEC=None, obs_time_str=None):
    Fcenter = 1420.405751
    freq_hz = np.linspace(Fcenter*1e6 - 1e6, Fcenter*1e6 + 1e6, nfft)
    diff_spec = avgps2 - avgps
    
    # Peak detection (For tangent point, ideally you want the edge, but keeping peak as requested)
    peak = np.argmax(diff_spec)
    obs_freq_mhz = freq_hz[peak] / 1e6

    # measured signal power
    peak_power = float(np.max(diff_spec))

    results = calculate_velocity_and_distance(obs_freq_mhz, L)

    if obs_time_str is None:
        obs_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.plot(freq_hz/1e6, avgps, 'b-', label='1422 MHz')
    plt.plot(freq_hz/1e6, avgps2, 'r--', label='1420 MHz')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power in counts')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.subplot(3, 1, 2)
    plt.plot(freq_hz/1e6, diff_spec, 'g-')
    plt.axvline(Fcenter, color='k', linestyle='--')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Signal Power')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.subplot(3, 1, 3)
    plt.axis('off')

    plt.text(0.5, 1.05, f"Observation Time: {obs_time_str}", fontsize=14, fontweight='bold', ha='center', va='bottom', transform=plt.gca().transAxes)

    # Prepare data for tables
    left_col = [ 
        ["Antenna Altitude", f"{alt:.2f}°"],
        ["Antenna Azimuth", f"{az:.2f}°"],
        ["Galactic Longitude (L)", f"{L:.2f}°"],
        ["Galactic Latitude (B)", f"{B:.2f}°"],
        ["Right Ascension (RA)", f"{RA:.2f}°"],
        
    ]
    
    # Modified Right Column for Tangent Point Method
    right_col = [
        ["Declination (DEC)", f"{DEC:.2f}°"],
        ["Observed Frequency", f"{obs_freq_mhz:.6f} MHz"],
        ["Radial Velocity (Vr)", f"{results['Vr']} km/s"],
        ["Tangent Point Distance (R)", f"{results['R']} kpc"],
        ["Rotation Velocity (V_rot)", f"{results['V_rot']} km/s"],
     
    ]

    table_left = plt.table(cellText=left_col, colWidths=[0.45, 0.45], loc='left', cellLoc='left', bbox=[0.01, 0.01, 0.48, 0.98])
    table_right = plt.table(cellText=right_col, colWidths=[0.45, 0.45], loc='right', cellLoc='left', bbox=[0.51, 0.01, 0.48, 0.98])
    table_left.auto_set_font_size(False)
    table_left.set_fontsize(10)
    table_right.auto_set_font_size(False)
    table_right.set_fontsize(10)

    plt.tight_layout()
    plot_time = obs_time_str.replace(":", ".").replace(" ", " at ")
    filename = f'plot_trial_{num}_{plot_time}.png'
    plt.savefig(filename, dpi=100)
    plt.close()

    return filename, freq_hz, results, obs_freq_mhz, peak_power

# --- MODIFIED: Save Data for Tangent Point ---
def save_data(avgps, avgps2, freq, results, trial_num, L, obs_freq, write_header=False, B=None, RA=None, DEC=None, filename=None, obs_time=None):
    if filename is None:
        filename = csv_filename
    os.makedirs("data", exist_ok=True)
    mode = 'w' if write_header else 'a'
    with open(filename, mode, newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Parameter", "Value", "Unit"])

        writer.writerow([f"Trial Number", trial_num, ""])
        writer.writerow(["Galactic Longitude (L)", L, "degrees"])
        if B is not None:
            writer.writerow(["Galactic Latitude (B)", B, "degrees"])
        if RA is not None:
            writer.writerow(["Right Ascension (RA)", RA, "degrees"])
        if DEC is not None:
            writer.writerow(["Declination (DEC)", DEC, "degrees"])
        writer.writerow(["observed Frequency", obs_freq, "MHz"])
        
        obs_time_val = obs_time if obs_time is not None else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow(["Observation Time", obs_time_val, ""])
        
        # Modified Parameters Section
        writer.writerow(["Calculated Parameters (Tangent Point Method)"])
        writer.writerow(["Radial Velocity (Vr)", results['Vr'], "km/s"])
        writer.writerow(["Tangent Point Distance (R)", results['R'], "kpc"])
        writer.writerow(["Rotation Velocity (V_rot)", results['V_rot'], "km/s"])
        
        writer.writerow(["Spectral Data"])
        writer.writerow(["Frequency (MHz)", "avgps", "avgps2"])

        for f_val, on, off in zip(freq/1e6, avgps.flatten(), avgps2.flatten()):
            writer.writerow([round(f_val, 4), round(on, 2), round(off, 2)])

        writer.writerow([])
        writer.writerow([])

# --- MODIFIED: Summary CSV for Tangent Point ---
def save_summary_csv(trial_num, obs_freq, results, L, B=None, RA=None, DEC=None, filename=None, obs_time=None, signal_power=None):
    if filename is None:
        filename = summary_filename
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.isfile(filename)
    obs_time_str = obs_time if obs_time is not None else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Updated Headers
            writer.writerow([
                "Trial Number", "Observation Time", "Signal Power", "Observed Frequency (MHz)", "Galactic Longitude (L)", 
                "Galactic Latitude (B)", "Right Ascension (RA)", "Declination (DEC)",
                "Radial Velocity Vr (km/s)", "Tangent Point Distance R (kpc)", "Rotation Velocity V_rot (km/s)"
            ])
        # Updated Row Data
        writer.writerow([
            trial_num, obs_time_str, (signal_power if signal_power is not None else ""), obs_freq, L,
            B if B is not None else "",
            RA if RA is not None else "",
            DEC if DEC is not None else "",
            results.get('Vr', ""), results.get('R', ""), results.get('V_rot', "")
        ])

# --- Streamlit Application Logic (Unchanged except imports/calls) ---
st.title('Hydrogen Line Observation')
st.markdown("""
This application processes radio telescope data to observe the 21cm hydrogen line 
and calculate galactic rotation parameters using the **Tangent Point Method**.
""")
plot_spot = st.empty()
num_samples = st.sidebar.number_input('Samples per Capture', min_value=512000, max_value=20480000, value=20480000, step=512000)
sample_rate = 2.4e6
num_trials = st.sidebar.number_input('Number of Trials', min_value=1, value=1)
nfft = st.sidebar.selectbox('FFT Size', [64, 128, 256, 512], index=1)

st.query_params.clear()

# Add this near the top, before coordinate calculation
st_autorefresh = st.sidebar.checkbox("Auto-refresh coordinates", value=True)
if st_autorefresh:
    st.experimental_rerun = st.experimental_rerun if hasattr(st, "experimental_rerun") else lambda: None
    st_autorefresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 60, 5)
    st_autorefresh_count = st.query_params.get("autorefresh_count", [0])
    st_autorefresh_count = int(st_autorefresh_count[0]) + 1
    import time
    if "last_autorefresh" not in st.session_state or \
       (datetime.now() - st.session_state.get("last_autorefresh", datetime.min)).total_seconds() > st_autorefresh_interval:
        st.session_state["last_autorefresh"] = datetime.now()
        st.query_params["autorefresh_count"] = st_autorefresh_count
        st.experimental_rerun()

def display_realtime_coordinates(alt_key="alt_slider", az_key="az_slider", show_sidebar=True):
    # Gauribidanur Radio Observatory coordinates
    latitude = 13.6029     # degrees North
    longitude = 77.4390    # degrees East
    elevation = 686.0      # meters

    if show_sidebar:
        st.sidebar.write(f"Latitude: {latitude}° N")
        st.sidebar.write(f"Longitude: {longitude}° E")
        st.sidebar.write(f"Elevation: {elevation} m")
        alt = st.sidebar.number_input("Antenna Altitude (°)", min_value=0.0, max_value=90.0, value=89.0, key=alt_key)
        az = st.sidebar.number_input("Antenna Azimuth (°)", min_value=0.0, max_value=360.0, value=180.0, key=az_key)
        # Sidebar live clock
        clock_placeholder = st.sidebar.empty()
        now = datetime.now()
        clock_placeholder.markdown(f"###  {now.strftime('%I:%M:%S %p')}")
        
        obs_time = Time(datetime.utcnow())
        obs_time_IST = obs_time + timedelta(hours=5, minutes=30)
        st.sidebar.write(obs_time_IST.strftime("%A, %B, %d, %Y, %I:%M%p IST"))
        
        location = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=elevation*u.m)
        altaz_frame = AltAz(obstime=obs_time, location=location)
        sky_coord = SkyCoord(alt=alt*u.deg, az=az*u.deg, frame=altaz_frame)
        icrs_coord = sky_coord.transform_to(ICRS())
        galactic_coord = sky_coord.transform_to(Galactic())
        
        L = galactic_coord.l.deg
        B = galactic_coord.b.deg
        RA = icrs_coord.ra.deg
        DEC = icrs_coord.dec.deg
        RA_hms = icrs_coord.ra.to_string(unit=u.hour, sep=':', precision=2, pad=True)
        
        st.sidebar.write(f"Alt: {alt:.2f}°, Az: {az:.2f}°")
        st.sidebar.write(f"Calculated Right Ascension: **{RA:.2f}°** ({RA_hms} hms)")
        st.sidebar.write(f"Calculated Declination: **{DEC:.2f}°**")
        st.sidebar.write(f"Calculated Galactic Latitude: **{B:.2f}°**")
        st.sidebar.write(f"Calculated Galactic Longitude: **{L:.2f}°**")
    else:
        # If a trial-specific slider key wasn't created, fall back to the global sliders
        # so plots and trials always use the current antenna pointing values.
        alt = st.session_state.get(alt_key, st.session_state.get('alt_slider', 89.0))
        az = st.session_state.get(az_key, st.session_state.get('az_slider', 180.0))
        obs_time = Time(datetime.utcnow())
        location = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=elevation*u.m)
        altaz_frame = AltAz(obstime=obs_time, location=location)
        sky_coord = SkyCoord(alt=alt*u.deg, az=az*u.deg, frame=altaz_frame)
        icrs_coord = sky_coord.transform_to(ICRS())
        galactic_coord = sky_coord.transform_to(Galactic())
        L = galactic_coord.l.deg
        B = galactic_coord.b.deg
        RA = icrs_coord.ra.deg
        DEC = icrs_coord.dec.deg
    
    return L, B, RA, DEC, alt, az

L, B, RA, DEC, alt, az = display_realtime_coordinates()
alt = st.session_state.get("alt_slider", 89.0)
az = st.session_state.get("az_slider", 180.0)

if st.button('Start Observation Session'):
    status_text = st.empty()
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)
    
    session_time = datetime.now().strftime("%Y-%m-%d started at %H.%M.%S")
    csv_filename = os.path.join(data_folder, f"trials_results_{session_time}.csv")
    summary_filename = os.path.join(data_folder, f"trials_summary_{session_time}.csv")

    # Re-define functions here to capture current filename/logic if needed, 
    # but the global definitions above are sufficient. 
    # We just call the global functions now.

    for trial in range(1, num_trials + 1):
        try:
            L, B, RA, DEC, trial_alt, trial_az = display_realtime_coordinates(
                alt_key=f"alt_slider_{trial}", az_key=f"az_slider_{trial}", show_sidebar=False
            )
            trial_alt = st.session_state.get(f"alt_slider_{trial}", trial_alt)
            trial_az = st.session_state.get(f"az_slider_{trial}", trial_az)

            status_text.text(f"Running Trial {trial}/{num_trials}...")
            aa = capture_rtl_data(1422000000, num_samples)
            bb = capture_rtl_data(1420405751, num_samples)
            avgps, avgps2 = process_data(aa, bb, nfft=nfft)

            obs_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            plot_filename, freq_hz, results, obs_freq, peak_power = create_plots(
                avgps, avgps2, trial, L, nfft, alt=trial_alt, az=trial_az, B=B, RA=RA, DEC=DEC, obs_time_str=obs_time_str
            )
            write_header = (trial == 1)
            save_data(avgps, avgps2, freq_hz, results, trial, L, obs_freq, write_header, B=B, RA=RA, DEC=DEC, filename=csv_filename, obs_time=obs_time_str)
            save_summary_csv(trial, obs_freq, results, L, B=B, RA=RA, DEC=DEC, filename=summary_filename, obs_time=obs_time_str, signal_power=peak_power)
            
            with plot_spot:
                st.image(plot_filename, caption=f'Trial {trial} Results')
            st.session_state.trial_num = trial
        except Exception as e:
            st.error(f"Error in trial {trial}: {str(e)}")
            break

    st.success("Observation session completed successfully!")