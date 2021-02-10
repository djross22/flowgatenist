# -*- coding: utf-8 -*-
"""
Class for defining metadata interface

To be eventually moved to a different package

@author: djross
"""

class MetaData():
    """
    Object containing metadata related to a flow cytometry measurement.
    
    Parameters
    ----------
    infile : str or file-like
        Reference to the associated FCS or raw data file.

    Attributes
    ----------
    infile : str or file-like
        Reference to associated FCS or raw data file.
    backfile : str or file-like
        Reference to associated background data file. The background data file
        should be in the same directory as the infile
    text : dict
        Dictionary of keyword-value entries from TEXT segment of the FCS
        file.
    analysis : dict
        Dictionary of keyword-value entries from ANALYSIS segment of the
        FCS file.
    data_type : str
        Type of data in the FCS file's DATA segment.
    time_step : float
        Time step of the time channel.
    acquisition_start_time : time or datetime
        Acquisition start time.
    acquisition_end_time : time or datetime
        Acquisition end time.
    acquisition_time : float
        Acquisition time, in seconds.
    channels : tuple
        The name of the channels contained in `FCSData`.
    scatter_back_fit : GaussianMixture
        The GMM fit to (log_fsc, log_ssc) from the background data
    scatter_cell_fit : GaussianMixture
        The GMM fit to (log_fsc, log_ssc) from the data (from infile)
        used to gate out likely background events, keeping only cell events
    scatter_singlet_fit : GaussianMixture
        The GMM fit to (SSC-W, adj_log_ssc) from the data (from infile)
        used to gate out multiple cell events, keeping only singlet events
    gmm_back_fraction : list of float
        Estimates of the number of background events that were kept after
        background gating, saved as a list, so if the fit is run more than
        once, the list will show how reproducible the estimate is.
    gmm_singlet_back_fraction : list of float
        Estimates of the number of background events that were kept after both
        background gating and singlet gating, saved as a list, so if the fit
        is run more than once, the list will show how reproducible the
        estimate is.
        
    bead_calibration_params : dictionary of functions
        Functions for conversion of fluorescence data into molecules of equilvalent fluor units.
        The dictionary keys are the fluorescence channel names.

    Methods
    -------
    amplification_type
        Get the amplification type used for the specified channel(s).
    detector_voltage
        Get the detector voltage used for the specified channel(s).
    amplifier_gain
        Get the amplifier gain used for the specified channel(s).
    range
        Get the range of the specified channel(s).
    resolution
        Get the resolution of the specified channel(s).
    hist_bins
        Get histogram bin edges for the specified channel(s).

    Notes
    -----


    """

    ###
    # Properties
    ###

    @property
    def background_signal(self):
        """
        Dictionary of background signal values

        """
        return self._background_signal

    @property
    def bead_calibration_params(self):
        """
        Dictionary of functions to convert from arb. fluorescence units to MEF
        #based on beads calibration.

        """
        return self._bead_calibration_params
    
    def bead_function(self, x, a, b):
        return a + b * x

    @property
    def gmm_singlet_back_fraction(self):
        """
        Reference to the estimated number of background events kept by the
        background subtraction and singlet gates.

        """
        return self._gmm_singlet_back_fraction

    @property
    def gmm_back_fraction(self):
        """
        Reference to the estimated number of background events kept by the
        background subtraction gate.

        """
        return self._gmm_back_fraction

    @property
    def scatter_singlet_fit(self):
        """
        Reference to the GMM fit to (SSC-W, adj_log_ssc)
        for singlet gating.

        """
        return self._scatter_singlet_fit

    @property
    def scatter_cell_fit(self):
        """
        Reference to the GMM fit to (log_fsc, log_ssc)
        for backghround subtraction.

        """
        return self._scatter_cell_fit

    @property
    def scatter_back_fit(self):
        """
        Reference to the GMM fit to (log_fsc, log_ssc)
        from the background data.

        """
        return self._scatter_back_fit

    @property
    def infile(self):
        """
        Reference to the associated FCS or raw data file.

        """
        return self._infile

    @property
    def backfile(self):
        """
        Reference to the associated FCS or raw data file.

        """
        return self._backfile

    @property
    def text(self):
        """
        Dictionary of key-value entries from the TEXT segment.

        `text` includes items from the TEXT segment and optional
        supplemental TEXT segment.

        """
        return self._text

    @property
    def analysis(self):
        """
        Dictionary of key-value entries from the ANALYSIS segment.

        """
        return self._analysis

    @property
    def data_type(self):
        """
        Type of data in the FCS file's DATA segment.

        `data_type` is 'I' if the data type is integer, 'F' for floating
        point, and 'D' for double.

        """
        return self._data_type

    @property
    def time_step(self):
        """
        Time step of the time channel.

        The time step is such that ``self[:,'Time']*time_step`` is in
        seconds. If no time step was found in the FCS file, `time_step` is
        None.

        """
        return self._time_step

    @property
    def acquisition_start_time(self):
        """
        Acquisition start time, as a python time or datetime object.

        `acquisition_start_time` is taken from the $BTIM keyword parameter
        in the TEXT segment of the FCS file. If date information is also
        found, `acquisition_start_time` is a datetime object with the
        acquisition date. If not, `acquisition_start_time` is a
        datetime.time object. If no start time is found in the FCS file,
        return None.

        """
        return self._acquisition_start_time

    @property
    def acquisition_end_time(self):
        """
        Acquisition end time, as a python time or datetime object.

        `acquisition_end_time` is taken from the $ETIM keyword parameter in
        the TEXT segment of the FCS file. If date information is also
        found, `acquisition_end_time` is a datetime object with the
        acquisition date. If not, `acquisition_end_time` is a datetime.time
        object. If no end time is found in the FCS file, return None.

        """
        return self._acquisition_end_time

    @property
    def acquisition_time(self):
        """
        Acquisition time, in seconds.

        The acquisition time is calculated using the 'time' channel by
        default (channel name is case independent). If the 'time' channel
        is not available, the acquisition_start_time and
        acquisition_end_time, extracted from the $BTIM and $ETIM keyword
        parameters will be used. If these are not found, None will be
        returned.

        """
        # Get time channels indices
        #time_channel_idx = [idx
        #                    for idx, channel in enumerate(self.channels)
        #                    if channel.lower() == 'time']
        #if len(time_channel_idx) > 1:
        #    raise KeyError("more than one time channel in data")
        ## Check if the time channel is available
        #elif len(time_channel_idx) == 1:
        ##    # Use the event list
        #    time_channel = self.channels[time_channel_idx[0]]
        #    return (self[-1, time_channel] - self[0, time_channel]) \
        #        * self.time_step
        #elif (self._acquisition_start_time is not None and
        #        self._acquisition_end_time is not None):
        #    # Use start_time and end_time:
        #    dt = (self._acquisition_end_time - self._acquisition_start_time)
        #    return dt.total_seconds()
        #else:
        #    return None
        
        dt = (self._acquisition_end_time - self._acquisition_start_time)
        
        return dt.total_seconds()

    @property
    def channels(self):
        """
        The name of the channels contained in `FCSData`.

        """
        return self._channels

    
