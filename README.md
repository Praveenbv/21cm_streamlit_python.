21cm Hydrogen Line using Python

Scripts used for radio astronomy.

*************tangent21.py*************

*The one you run while the antenna is pointed at the sky. Launch it with:

"""""streamlit run tangent21.py"""""


>>Streamlit app, so it runs in the browser
>>Set your antenna's Alt/Az in the sidebar and hit "Start Observation Session"
>>Captures data at 1420.405751 MHz (the hydrogen line) and 1422 MHz (off-line reference, used to subtract out the background)
>>Takes the difference of the two, finds the peak, works out radial velocity
>>Uses the tangent point method to estimate distance and rotation speed of that gas
>>Converts your Alt/Az into RA/Dec and galactic coordinates automatically (astropy)
>>Location is hardcoded for Gauribidanur Radio Observatory - change lat/lon/elevation in display_realtime_coordinates() if you're elsewhere
>>Saves a plot per trial (spectra + results table) as a PNG
>>Writes two CSVs into data/ - full spectral dump, and a one-line-per-trial summary
>>Can run multiple trials per session
>>Has an auto-refresh option for the sidebar coordinates

>>>> By default, the location coordinates are set to the Gauribidanur Radio Observatory itself.
>>>> It can be changed in the code function called


"""""""""""""""""""

************************display_realtime_coordinates(alt_key="alt_slider", az_key="az_slider", show_sidebar=True):
    # Gauribidanur Radio Observatory coordinates
    latitude = 13.6029     # degrees North
    longitude = 77.4390    # degrees East
    elevation = 686.0      # meters...........................................................**********************************


                                                                                         """"""""""""""""""""""""""""""

**********sky_map_plotter_with_summery_data.py***************

To run this, use the command below comannd

>>python sky_map_plotter_with_summery_data.py


>>Looks for a zip named gbd_horn_All_drift_scan_data_summary.zip and extracts it
>>Falls back to globbing for *summary*.csv files if there's no zip
>>Combines all the RA/Dec/signal power data
>>Interpolates it onto a grid with griddata
>>Runs a destriping step - lines up each declination row to a common background level, gets rid of the horizontal banding and bright zenith bump that from drift scan
>>Smooths with a Gaussian filter
>>Plots the result with pcolormesh, full 360 degrees of RA


***********requirements**********

pip install streamlit numpy matplotlib scipy pandas astropy pyrtlsdr


**An RTL-SDR dongle
**An antenna that can pick up 1420 MHz (horn antenna is the usual choice)
**librtlsdr installed on your system, or pyrtlsdr has nothing to talk to


***quick note on the tangent point method***

Looking along a line of sight into the inner galaxy (galactic longitude between 0 and 90 degrees), there's a point - the tangent point - which is as close as that line ever gets to the galactic centre. Gas there has the highest radial velocity of anything along that sightline, which is what makes it useful:


Distance: R = R0 * sin(L)
Rotation velocity: V_rot = Vr + V0 * sin(L)
R0 (8.5 kpc) and V0 (220 km/s) are the assumed distance and orbital speed of the Sun around the Galactic Centre
Vr comes from the Doppler shift measured on the line itself

Peak finding is just argmax on the difference spectrum - fine for a strong, clean signal, not great once things get noisy 
Destriping only handles horizontal banding, not anything RA-dependent
There's a workaround for st.experimental_rerun since Streamlit deprecated it - should switch to st.rerun() at some point
Observatory coordinates and the hydrogen rest frequency are  constants near the top of the files.
*****note****: Antenna pointing and entering coordinates in the Streamlit web page is very important; if this is wrong, everything goes wrong.
