"""
Classes and utiliy functions for reading FCS files.

Most of the code in this file was copied from FlowCal version 1.1.4:
    https://github.com/taborlab/FlowCal

The license for that code included the following:
    The MIT License (MIT)
    
    Copyright (c) 2015 John Sexton, Brian Landry, Sebastian Castillo-Hair
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
"""

import os
import copy
import collections
import datetime
import warnings
import pickle

import numpy as np
import pandas as pd

import flowgatenist.metadata
import flowgatenist.transform

###
# Utility functions for importing segments of FCS files
###

def read_fcs_header_segment(buf, begin=0):
    """
    Read HEADER segment of FCS file.

    Parameters
    ----------
    buf : file-like object
        Buffer containing data to interpret as HEADER segment.
    begin : int
        Offset (in bytes) to first byte of HEADER segment in `buf`.

    Returns
    -------
    header : namedtuple
        Version information and byte offset values of other FCS segments
        (see FCS standards for more information) in the following order:
            - version : str
            - text_begin : int
            - text_end : int
            - data_begin : int
            - data_end : int
            - analysis_begin : int
            - analysis_end : int

    Notes
    -----
    Blank ANALYSIS segment offsets are converted to zeros.

    OTHER segment offsets are ignored (see [1]_, [2]_, and [3]_).

    References
    ----------
    .. [1] P.N. Dean, C.B. Bagwell, T. Lindmo, R.F. Murphy, G.C. Salzman,
       "Data file standard for flow cytometry. Data File Standards
       Committee of the Society for Analytical Cytology," Cytometry vol
       11, pp 323-332, 1990, PMID 2340769.

    .. [2] L.C. Seamer, C.B. Bagwell, L. Barden, D. Redelman, G.C. Salzman,
       J.C. Wood, R.F. Murphy, "Proposed new data file standard for flow
       cytometry, version FCS 3.0," Cytometry vol 28, pp 118-122, 1997,
       PMID 9181300.

    .. [3] J. Spidlen, et al, "Data File Standard for Flow Cytometry,
       version FCS 3.1," Cytometry A vol 77A, pp 97-100, 2009, PMID
       19937951.

    """
    fields = [
        'version',
        'text_begin',
        'text_end',
        'data_begin',
        'data_end',
        'analysis_begin',
        'analysis_end']

    FCSHeader = collections.namedtuple('FCSHeader', fields)

    field_values = []

    buf.seek(begin)
    # decode is for Python version 3:
    field_values.append(buf.read(10).decode('ascii').rstrip())     # version
    # field_values.append(str(buf.read(10)).rstrip())     # version

    field_values.append(int(buf.read(8)))               # text_begin
    field_values.append(int(buf.read(8)))               # text_end
    field_values.append(int(buf.read(8)))               # data_begin
    field_values.append(int(buf.read(8)))               # data_end

    fv = buf.read(8)                                    # analysis_begin
    field_values.append(0 if fv == ' '*8 else int(fv))
    fv = buf.read(8)                                    # analysis_end
    field_values.append(0 if fv == ' '*8 else int(fv))

    header = FCSHeader._make(field_values)
    return header

def read_fcs_text_segment(buf, begin, end, delim=None):
    """
    Read TEXT segment of FCS file.

    Parameters
    ----------
    buf : file-like object
        Buffer containing data to interpret as TEXT segment.
    begin : int
        Offset (in bytes) to first byte of TEXT segment in `buf`.
    end : int
        Offset (in bytes) to last byte of TEXT segment in `buf`.
    delim : str, optional
        1-byte delimiter character which delimits key-value entries of
        TEXT segment. If None, will extract delimter as first byte
        of TEXT segment.

    Returns
    -------
    text : dict
        Dictionary of key-value entries extracted from TEXT segment.
    delim : str
        String containing delimiter character.

    Raises
    ------
    ValueError
        If TEXT segment does not start and end with delimiter.
    ValueError
        If function detects odd number of total extracted keys and
        values (indicating an unpaired key or value).
    NotImplementedError
        If delimiter is used in a keyword or value.

    Notes
    -----
    ANALYSIS segments and TEXT segments are parsed the same way, so
    this function can also be used to parse ANALYSIS segments.

    This function does not automatically parse supplemental TEXT
    segments (see FCS3.0 [2] and FCS3.1 [3]). Supplemental TEXT segments
    and regular TEXT segments are parsed the same way, though, so this
    function can be manually directed to parse a supplemental TEXT segment
    by providing the appropriate `begin` and `end` values.

    References
    ----------
    .. [1] P.N. Dean, C.B. Bagwell, T. Lindmo, R.F. Murphy, G.C. Salzman,
       "Data file standard for flow cytometry. Data File Standards
       Committee of the Society for Analytical Cytology," Cytometry vol
       11, pp 323-332, 1990, PMID 2340769.

    .. [2] L.C. Seamer, C.B. Bagwell, L. Barden, D. Redelman, G.C. Salzman,
       J.C. Wood, R.F. Murphy, "Proposed new data file standard for flow
       cytometry, version FCS 3.0," Cytometry vol 28, pp 118-122, 1997,
       PMID 9181300.

    .. [3] J. Spidlen, et al, "Data File Standard for Flow Cytometry,
       version FCS 3.1," Cytometry A vol 77A, pp 97-100, 2009, PMID
       19937951.

    """
    if delim is None:
        buf.seek(begin)
        #delim = str(buf.read(1))
        delim = buf.read(1)

    # The offsets are inclusive (meaning they specify first and last byte
    # WITHIN segment) and seeking is inclusive (read() after seek() reads the
    # byte which was seeked to). This means the length of the segment is
    # ((end+1) - begin).
    buf.seek(begin)
    raw = buf.read((end+1)-begin)

    # If segment is empty, return empty dictionary as text
    if not raw:
        return {}, delim

    # Check that the first character of the TEXT segment is equal to the
    # delimiter.
    if raw[0:1] != delim:
        raise ValueError("segment should start with delimiter")

    # Look for the last delimiter in the segment string, and retain everything
    # from one character after the first delimiter to one character before the
    # last delimiter.
    end_index = raw.rfind(delim)
    raw = raw[1: end_index]

    pairs_list = raw.split(delim)
    string_pairs_list = []
    for pa in pairs_list:
        string_pairs_list.append(pa.decode('UTF-8'))
    pairs_list = string_pairs_list

    # According to the FCS2.0 standard, "If the separator appears in a keyword
    # or in a keyword value, it must be 'quoted' by being repeated" and "null
    # (zero length) keywords or keyword values are not permitted", so this
    # issue should manifest itself as an empty element in the list.
    # The following scans the list of pairs for empty elements and appends a
    # delimiter character to the previous element when an empty element is
    # found.
    pairs_list_delim = []
    pairs_list_idx = 0
    while pairs_list_idx < len(pairs_list):
        if pairs_list[pairs_list_idx] != '':
            # Non-empty element, just append
            pairs_list_delim.append(pairs_list[pairs_list_idx])
        else:
            # Empty element
            # Accumulate delimiters in a temporary string as long as more empty
            # elements are found, until the last element
            s = ''
            while True:
                s += str(delim)
                pairs_list_idx += 1
                if pairs_list_idx >= len(pairs_list):
                    break
                if pairs_list[pairs_list_idx] != '':
                    s += pairs_list[pairs_list_idx]
                    break
            # Append temporary string to previous element if pairs_list_delim
            # is not empty, otherwise make it the first element.
            if pairs_list_delim:
                pairs_list_delim[-1] += s
            else:
                pairs_list_delim.append(s)
        pairs_list_idx += 1
    pairs_list = pairs_list_delim

    # List length should be even since all key-value entries should be pairs
    if len(pairs_list) % 2 != 0:
        raise ValueError("odd # of (keys + values); unpaired key or value")

    text = dict(list(zip(pairs_list[0::2], pairs_list[1::2])))

    return text, delim

def read_fcs_data_segment(buf,
                          begin,
                          end,
                          datatype,
                          num_events,
                          param_bit_widths,
                          big_endian,
                          param_ranges=None):
    """
    Read DATA segment of FCS file.

    Parameters
    ----------
    buf : file-like object
        Buffer containing data to interpret as DATA segment.
    begin : int
        Offset (in bytes) to first byte of DATA segment in `buf`.
    end : int
        Offset (in bytes) to last byte of DATA segment in `buf`.
    datatype : {'I', 'F', 'D', 'A'}
        String specifying FCS file datatype (see $DATATYPE keyword from
        FCS standards). Supported datatypes include 'I' (unsigned
        binary integer), 'F' (single precision floating point), and 'D'
        (double precision floating point). 'A' (ASCII) is recognized
        but not supported.
    num_events : int
        Total number of events (see $TOT keyword from FCS standards).
    param_bit_widths : array-like
        Array specifying parameter (aka channel) bit width for each
        parameter (see $PnB keywords from FCS standards). The length of
        `param_bit_widths` should match the $PAR keyword value from the
        FCS standards (which indicates the total number of parameters).
        If `datatype` is 'I', data must be byte aligned (i.e. all
        parameter bit widths should be divisible by 8), and data are
        upcast to the nearest uint8, uint16, uint32, or uint64 data
        type. Bit widths larger than 64 bits are not supported.
    big_endian : bool
        Endianness of computer used to acquire data (see $BYTEORD
        keyword from FCS standards). True implies big endian; False
        implies little endian.
    param_ranges : array-like, optional
        Array specifying parameter (aka channel) range for each
        parameter (see $PnR keywords from FCS standards). Used to
        ensure erroneous values are not read from DATA segment by
        applying a bit mask to remove unused bits. The length of
        `param_ranges` should match the $PAR keyword value from the FCS
        standards (which indicates the total number of parameters). If
        None, no masking is performed.

    Returns
    -------
    data : numpy array
        NxD numpy array describing N cytometry events observing D data
        dimensions.

    Raises
    ------
    ValueError
        If lengths of `param_bit_widths` and `param_ranges` don't match.
    ValueError
        If calculated DATA segment size (as determined from the number
        of events, the number of parameters, and the number of bytes per
        data point) does not match size specified by `begin` and `end`.
    ValueError
        If `param_bit_widths` doesn't agree with `datatype` for single
        precision or double precision floating point (i.e. they should
        all be 32 or 64, respectively).
    ValueError
        If `datatype` is unrecognized.
    NotImplementedError
        If `datatype` is 'A'.
    NotImplementedError
        If `datatype` is 'I' but data is not byte aligned.

    References
    ----------
    .. [1] P.N. Dean, C.B. Bagwell, T. Lindmo, R.F. Murphy, G.C. Salzman,
       "Data file standard for flow cytometry. Data File Standards
       Committee of the Society for Analytical Cytology," Cytometry vol
       11, pp 323-332, 1990, PMID 2340769.

    .. [2] L.C. Seamer, C.B. Bagwell, L. Barden, D. Redelman, G.C. Salzman,
       J.C. Wood, R.F. Murphy, "Proposed new data file standard for flow
       cytometry, version FCS 3.0," Cytometry vol 28, pp 118-122, 1997,
       PMID 9181300.

    .. [3] J. Spidlen, et al, "Data File Standard for Flow Cytometry,
       version FCS 3.1," Cytometry A vol 77A, pp 97-100, 2009, PMID
       19937951.

    """
    num_params = len(param_bit_widths)
    if (param_ranges is not None and len(param_ranges) != num_params):
        raise ValueError("param_bit_widths and param_ranges must have same"
            + " length")

    shape = (int(num_events), num_params)

    if datatype == 'I':
        # Check if all parameters fit into preexisting data type
        if (all(bw == 8  for bw in param_bit_widths) or
            all(bw == 16 for bw in param_bit_widths) or
            all(bw == 32 for bw in param_bit_widths) or
            all(bw == 64 for bw in param_bit_widths)):

            num_bits = param_bit_widths[0]

            # Sanity check that the total # of bytes that we're about to
            # interpret is exactly the # of bytes in the DATA segment.
            # In some FCS files, the offset to the last byte (end) actually
            # points to the first byte of the next segment, in which case the #
            # of bytes specified in the header exceeds the # of bytes that we
            # should read by one.
            if (shape[0]*shape[1]*(num_bits//8)) != ((end+1)-begin) and \
                    (shape[0]*shape[1]*(num_bits//8)) != (end-begin):
                raise ValueError("DATA size does not match expected array"
                    + " size (array size ="
                    + " {0} bytes,".format(shape[0]*shape[1]*(num_bits//8))
                    + " DATA segment size = {0} bytes)".format((end+1)-begin))

            dtype = np.dtype('{0}u{1}'.format('>' if big_endian else '<',
                                              num_bits//8))
            data = np.memmap(
                buf,
                dtype=dtype,
                mode='r',
                offset=begin,
                shape=shape,
                order='C')

            # Cast memmap object to regular numpy array stored in memory (as
            # opposed to being backed by disk)
            data = np.array(data)
        else:
            # The FCS standards technically allows for parameters to NOT be
            # byte aligned, but parsing a DATA segment which is not byte
            # aligned requires significantly more computation (and probably an
            # external library which exposes bit level resolution to a block
            # of memory). I don't think this is a common use case, so I'm just
            # going to detect it and raise an error.
            if (not all(bw % 8 == 0 for bw in param_bit_widths) or
                any(bw > 64 for bw in param_bit_widths)):
                raise NotImplementedError("only byte aligned parameter bit"
                    + " widths (bw % 8 = 0) <= 64 are supported"
                    + " (param_bit_widths={0})".format(param_bit_widths))

            # Read data in as a byte array
            byte_shape = (int(num_events),
                          np.sum(np.array(param_bit_widths)//8))

            # Sanity check that the total # of bytes that we're about to
            # interpret is exactly the # of bytes in the DATA segment.
            # In some FCS files, the offset to the last byte (end) actually
            # points to the first byte of the next segment, in which case the #
            # of bytes specified in the header exceeds the # of bytes that we
            # should read by one.
            if (byte_shape[0]*byte_shape[1]) != ((end+1)-begin) and \
                    (byte_shape[0]*byte_shape[1]) != (end-begin):
                raise ValueError("DATA size does not match expected array"
                    + " size (array size ="
                    + " {0} bytes,".format(byte_shape[0]*byte_shape[1])
                    + " DATA segment size = {0} bytes)".format((end+1)-begin))

            byte_data = np.memmap(
                buf,
                dtype='uint8',  # endianness doesn't matter for 1 byte
                mode='r',
                offset=begin,
                shape=byte_shape,
                order='C')

            # Upcast all data to fit nearest supported data type of largest
            # bit width
            upcast_bw = int(2**np.max(np.ceil(np.log2(param_bit_widths))))

            # Create new array of upcast data type and use byte data to
            # populate it. The new array will have endianness native to user's
            # machine; does not preserve endianness of stored FCS data.
            upcast_dtype = 'u{0}'.format(upcast_bw//8)
            data = np.zeros(shape,dtype=upcast_dtype)

            # Array mapping each column of data to first corresponding column
            # in byte_data
            byte_boundaries = np.roll(np.cumsum(param_bit_widths)//8,1)
            byte_boundaries[0] = 0

            # Reconstitute columns of data by bit shifting appropriate columns
            # in byte_data and accumulating them
            for col in range(data.shape[1]):
                num_bytes = param_bit_widths[col]//8
                for b in range(num_bytes):
                    byte_data_col = byte_boundaries[col] + b
                    byteshift = (num_bytes-b-1) if big_endian else b

                    if byteshift > 0:
                        # byte_data must be upcast or else bit shift fails
                        data[:,col] += \
                            byte_data[:,byte_data_col].astype(upcast_dtype) \
                            << (byteshift*8)
                    else:
                        data[:,col] += byte_data[:,byte_data_col]

        if param_ranges is not None:
            # To strictly follow the FCS standards, mask off the unused high bits
            # as specified by param_ranges.
            for col in range(data.shape[1]):
                # bits_used should be related to resolution of cytometer ADC
                bits_used = int(np.ceil(np.log2(param_ranges[col])))

                # Create a bit mask to mask off all but the lowest bits_used bits.
                # bitmask is a native python int type which does not have an
                # underlying size. The int type is effectively left-padded with
                # 0s (infinitely), and the '&' operation preserves the dataype of
                # the array, so this shouldn't be an issue.
                bitmask = ~((~0) << bits_used)
                data[:,col] &= bitmask

    elif datatype in ('F','D'):
        num_bits = 32 if datatype == 'F' else 64

        # Confirm that bit widths are consistent with data type
        if not all(bw == num_bits for bw in param_bit_widths):
            raise ValueError("all param_bit_widths should be"
                + " {0} if datatype =".format(num_bits)
                + " \'{0}\' (param_bit_widths=".format(datatype)
                + "{0})".format(param_bit_widths))

        # Sanity check that the total # of bytes that we're about to interpret
        # is exactly the # of bytes in the DATA segment.
        # In some FCS files, the offset to the last byte (end) actually points
        # to the first byte of the next segment, in which case the # of bytes
        # specified in the header exceeds the # of bytes that we should read by
        # one.
        if (shape[0]*shape[1]*(num_bits//8)) != ((end+1)-begin) and \
            (shape[0]*shape[1]*(num_bits//8)) != (end-begin):
            raise ValueError("DATA size does not match expected array size"
                + " (array size = {0}".format(shape[0]*shape[1]*(num_bits//8))
                + " bytes, DATA segment size ="
                + " {0} bytes)".format((end+1)-begin))

        dtype = np.dtype('{0}f{1}'.format('>' if big_endian else '<',
                                          num_bits//8))
        data = np.memmap(
            buf,
            dtype=dtype,
            mode='r',
            offset=begin,
            shape=shape,
            order='C')

        # Cast memmap object to regular numpy array stored in memory (as
        # opposed to being backed by disk)
        data = np.array(data)
    elif datatype == 'A':
        raise NotImplementedError("only \'I\' (unsigned binary integer),"
            + " \'F\' (single precision floating point), and \'D\' (double"
            + " precision floating point) data types are supported (detected"
            + " datatype=\'{0}\')".format(datatype))
    else:
        raise ValueError("unrecognized datatype (detected datatype="
            + "\'{0}\')".format(datatype))

    return data

###
# Classes
###

class FCSFile(object):
    """
    Class representing an FCS flow cytometry data file.

    This class parses a binary FCS file and exposes a read-only view
    of the HEADER, TEXT, DATA, and ANALYSIS segments via Python-friendly
    data structures.

    Parameters
    ----------
    infile : str or file-like
        Reference to the associated FCS file.

    Attributes
    ----------
    infile : str or file-like
        Reference to associated FCS file.
    header : namedtuple
        Version information and byte offset values of other FCS segments
        in the following order:
            - version : str
            - text_begin : int
            - text_end : int
            - data_begin : int
            - data_end : int
            - analysis_begin : int
            - analysis_end : int
    text : dict
        Dictionary of keyword-value entries from TEXT segment and optional
        supplemental TEXT segment.
    data : numpy array
        Unwriteable NxD numpy array describing N cytometry events
        observing D data dimensions.
    analysis : dict
        Dictionary of keyword-value entries from ANALYSIS segment.

    Raises
    ------
    NotImplementedError
        If $MODE is not 'L'.
    NotImplementedError
        If $DATATYPE is not 'I', 'F', or 'D'.
    NotImplementedError
        If $DATATYPE is 'I' but data is not byte aligned.
    NotImplementedError
        If $BYTEORD is not big endian ('4,3,2,1' or '2,1') or little
        endian ('1,2,3,4', '1,2').
    ValueError
        If TEXT-like segment does not start with delimiter.
    ValueError
        If TEXT-like segment has odd number of total extracted keys and
        values (indicating an unpaired key or value).
    ValueError
        If calculated DATA segment size (as determined from the number
        of events, the number of parameters, and the number of bytes per
        data point) does not match size specified in HEADER segment
        offsets.
    Warning
        If more than one data set is detected in the same file.
    Warning
        If the ANALYSIS segment was not successfully parsed.
    
    Notes
    -----
    The Flow Cytometry Standard (FCS) describes the de facto standard
    file format used by flow cytometry acquisition and analysis software
    to record flow cytometry data to and load flow cytometry data from a
    file. The standard dictates that each file must have the following
    segments: HEADER, TEXT, and DATA. The HEADER segment contains
    version information and byte offset values of other segments, the
    TEXT segment contains delimited key-value pairs containing
    acquisition information, and the DATA segment contains the recorded
    flow cytometry data. The file may optionally have an ANALYSIS
    segment (structurally identicaly to the TEXT segment), a
    supplemental TEXT segment (according to more recent versions of the
    standard), and user-defined OTHER segments.

    This class supports a subset of the FCS3.1 standard which should be
    backwards compatible with FCS3.0 and FCS2.0. The FCS file must be
    of the following form:
        - $MODE = 'L' (list mode; histogram mode is not supported).
        - $DATATYPE = 'I' (unsigned binary integers), 'F' (single
          precision floating point), or 'D' (double precision floating
          point). 'A' (ASCII) is not supported.
        - If $DATATYPE = 'I', $PnB % 8 = 0 (byte aligned) for all
          parameters (aka channels).
        - $BYTEORD = '4,3,2,1' (big endian) or '1,2,3,4' (little
          endian).
        - One data set per file.

    For more information on the TEXT segment keywords (e.g. $MODE,
    $DATATYPE, etc.), see [1]_, [2]_, and [3]_.

    References
    ----------
    .. [1] P.N. Dean, C.B. Bagwell, T. Lindmo, R.F. Murphy, G.C. Salzman,
       "Data file standard for flow cytometry. Data File Standards
       Committee of the Society for Analytical Cytology," Cytometry vol
       11, pp 323-332, 1990, PMID 2340769.

    .. [2] L.C. Seamer, C.B. Bagwell, L. Barden, D. Redelman, G.C. Salzman,
       J.C. Wood, R.F. Murphy, "Proposed new data file standard for flow
       cytometry, version FCS 3.0," Cytometry vol 28, pp 118-122, 1997,
       PMID 9181300.

    .. [3] J. Spidlen, et al, "Data File Standard for Flow Cytometry,
       version FCS 3.1," Cytometry A vol 77A, pp 97-100, 2009, PMID
       19937951.

    """
    def __init__(self, infile):
        
        self._infile = infile

        if isinstance(infile, str):
            f = open(infile, 'rb')
        else:
            f = infile

        self._header = read_fcs_header_segment(buf=f)

        # Import primary TEXT segment and optional supplemental TEXT segment.
        # Primary TEXT segment offsets are always specified in the HEADER
        # segment. For FCS3.0 and above, supplemental TEXT segment offsets
        # are always specified via required key-value pairs in the primary
        # TEXT segment.
        self._text, delim = read_fcs_text_segment(
            buf=f,
            begin=self._header.text_begin,
            end=self._header.text_end)

        if self._header.version in ('FCS3.0','FCS3.1'):
            stext_begin = int(self._text['$BEGINSTEXT'])   # required keyword
            stext_end = int(self._text['$ENDSTEXT'])       # required keyword
            if stext_begin and stext_end:
                stext = read_fcs_text_segment(
                    buf=f,
                    begin=stext_begin,
                    end=stext_end,
                    delim=delim)[0]
                self._text.update(stext)

        # Confirm FCS file assumptions. All queried keywords are required
        # keywords.
        if self._text['$MODE'] != 'L':
            raise NotImplementedError("only $MODE = \'L\' is supported"
                + " (detected $MODE = \'{0}\')".format(self._text['$MODE']))

        if self._text['$DATATYPE'] not in ('I','F','D'):
            raise NotImplementedError("only $DATATYPE = \'I\', \'F\', and"
                + " \'D\' are supported (detected $DATATYPE ="
                + " \'{0}\')".format(self._text['$DATATYPE']))

        D = int(self._text['$PAR']) # total number of parameters (aka channels)
        param_bit_widths = [int(self._text['$P{0}B'.format(p)])
                            for p in range(1,D+1)]
        if self._text['$DATATYPE'] == 'I':
            if not all(bw % 8 == 0 for bw in param_bit_widths):
                raise NotImplementedError("if $DATATYPE = \'I\', only byte"
                    + " aligned parameter bit widths (bw % 8 = 0) are"
                    + " supported (detected {0})".format(
                        ", ".join('$P{0}B={1}'.format(
                            p,self._text['$P{0}B'.format(p)])
                        for p in range(1,D+1)
                        if param_bit_widths[p-1] % 8 != 0)))

        if self._text['$BYTEORD'] not in ('4,3,2,1', '2,1', '1,2,3,4', '1,2'):
            raise NotImplementedError("only big endian ($BYTEORD = \'4,3,2,1\'"
                + " or \'2,1\') and little endian ($BYTEORD = \'1,2,3,4\' or"
                + " \'1,2\') are supported (detected $BYTEORD ="
                + " \'{0}\')".format(self._text['$BYTEORD']))
        big_endian = self._text['$BYTEORD'] in ('4,3,2,1', '2,1')

        if int(self._text['$NEXTDATA']):
            warnings.warn("detected (and ignoring) additional data set"
                + " ($NEXTDATA = {0})".format(self._text['$NEXTDATA']))

        # Import optional ANALYSIS segment
        if self._header.analysis_begin and self._header.analysis_end:
            # Prioritize ANALYSIS segment offsets specified in HEADER over
            # offsets specified in TEXT segment.
            try:
                self._analysis = read_fcs_text_segment(
                    buf=f,
                    begin=self._header.analysis_begin,
                    end=self._header.analysis_end,
                    delim=delim)[0]
            except Exception as e:
                warnings.warn("ANALYSIS segment could not be parsed ({})".\
                    format(str(e)))
                self._analysis = {}
        elif self._header.version in ('FCS3.0', 'FCS3.1'):
            analysis_begin = int(self._text['$BEGINANALYSIS'])
            analysis_end = int(self._text['$ENDANALYSIS'])
            if analysis_begin and analysis_end:
                try:
                    self._analysis = read_fcs_text_segment(
                        buf=f,
                        begin=analysis_begin,
                        end=analysis_end,
                        delim=delim)[0]
                except Exception as e:
                    warnings.warn("ANALYSIS segment could not be parsed ({})".\
                        format(str(e)))
                    self._analysis = {}
            else:
                self._analysis = {}
        else:
            self._analysis = {}
        
        # Import DATA segment
        param_ranges = [float(self._text['$P{0}R'.format(p)])
                        for p in range(1,D+1)]
        if self._header.data_begin and self._header.data_end:
            # Prioritize DATA segment offsets specified in HEADER over
            # offsets specified in TEXT segment.
            self._data = read_fcs_data_segment(
                buf=f,
                begin=self._header.data_begin,
                end=self._header.data_end,
                datatype=self._text['$DATATYPE'],
                num_events=int(self._text['$TOT']),
                param_bit_widths=param_bit_widths,
                param_ranges=param_ranges,
                big_endian=big_endian)
        elif self._header.version in ('FCS3.0', 'FCS3.1'):
            data_begin = int(self._text['$BEGINDATA'])
            data_end = int(self._text['$ENDDATA'])
            if data_begin and data_end:
                self._data = read_fcs_data_segment(
                    buf=f,
                    begin=data_begin,
                    end=data_end,
                    datatype=self._text['$DATATYPE'],
                    num_events=int(self._text['$TOT']),
                    param_bit_widths=param_bit_widths,
                    param_ranges=param_ranges,
                    big_endian=big_endian)
            else:
                raise ValueError("DATA segment incorrectly specified")
        else:
            raise ValueError("DATA segment incorrectly specified")
        self._data.flags.writeable = False

        if isinstance(infile, str):
            f.close()

    # Expose attributes as read-only properties
    @property
    def infile(self):
        """
        Reference to the associated FCS file.

        """
        return self._infile

    @property
    def header(self):
        """
        ``namedtuple`` containing version information and byte offset
        values of other FCS segments in the following order:
            - version : str
            - text_begin : int
            - text_end : int
            - data_begin : int
            - data_end : int
            - analysis_begin : int
            - analysis_end : int

        """
        return self._header

    @property
    def text(self):
        """
        Dictionary of key-value entries from TEXT segment and optional
        supplemental TEXT segment.

        """
        return self._text

    @property
    def data(self):
        """
        Unwriteable NxD numpy array describing N cytometry events
        observing D data dimensions.

        """
        return self._data

    @property
    def analysis(self):
        """
        Dictionary of key-value entries from ANALYSIS segment.

        """
        return self._analysis

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.infile == other.infile
                and self.header == other.header
                and self.text == other.text
                and np.array_equal(self.data, other.data)
                and self.analysis == other.analysis)
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self == other
        else:
            return NotImplemented

    def __hash__(self):
        return hash((self.infile,
                     self.header,
                     frozenset(list(self.text.items())),
                     self.data.tobytes(),
                     frozenset(list(self.analysis.items()))))

    def __repr__(self):
        return str(self.infile)

class FCSData(np.ndarray):
    """
    Object containing events data from a flow cytometry sample.

    An `FCSData` object is an NxD numpy array representing N cytometry
    events with D dimensions (channels) extracted from the DATA segment of
    an FCS file. Indexing along the second axis can be performed by channel
    name, which allows to easily select data from one or several channels.
    Otherwise, an `FCSData` object can be treated as a numpy array for most
    purposes.

    Information regarding the acquisition date, time, and information about
    the detector and the amplifiers are parsed from the TEXT segment of the
    FCS file and exposed as attributes. The TEXT and ANALYSIS segments are
    also exposed as attributes.
    
    Parameters
    ----------
    infile : str or file-like
        Reference to the associated FCS file.

    Attributes
    ----------
    infile : str or file-like
        Reference to associated FCS file.
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
    `FCSData` uses `FCSFile` to parse an FCS file. All restrictions on the
    FCS file format and the Exceptions spcecified for FCSFile also apply
    to FCSData.

    Parsing of some non-standard files is supported [4]_.

    References
    ----------
    .. [1] P.N. Dean, C.B. Bagwell, T. Lindmo, R.F. Murphy, G.C. Salzman,
       "Data file standard for flow cytometry. Data File Standards
       Committee of the Society for Analytical Cytology," Cytometry vol
       11, pp 323-332, 1990, PMID 2340769.

    .. [2] L.C. Seamer, C.B. Bagwell, L. Barden, D. Redelman, G.C. Salzman,
       J.C. Wood, R.F. Murphy, "Proposed new data file standard for flow
       cytometry, version FCS 3.0," Cytometry vol 28, pp 118-122, 1997,
       PMID 9181300.

    .. [3] J. Spidlen, et al, "Data File Standard for Flow Cytometry,
       version FCS 3.1," Cytometry A vol 77A, pp 97-100, 2009, PMID
       19937951.

    .. [4] R. Hicks, "BD$WORD file header fields,"
       https://lists.purdue.edu/pipermail/cytometry/2001-October/020624.html

    Examples
    --------
    Load an FCS file into an FCSData object

    >>> import flowgatenist
    >>> d = flowgatenist.io.FCSData('test/Data001.fcs')

    Check channel names

    >>> print d.channels
    ('FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time')

    Check the size of FCSData

    >>> print d.shape
    (20949, 6)

    Get the first 100 events

    >>> d_sub = d[:100]
    >>> print d_sub.shape
    (100, 6)

    Retain only fluorescence channels

    >>> d_fl = d[:, ['FL1-H', 'FL2-H', 'FL3-H']]
    >>> d_fl.channels
    ('FL1-H', 'FL2-H', 'FL3-H')

    Channel slicing can also be done with integer indices

    >>> d_fl_2 = d[:, [2, 3, 4]]
    >>> print d_fl_2.channels
    ('FL1-H', 'FL2-H', 'FL3-H')
    >>> import numpy as np
    >>> np.all(d_fl == d_fl_2)
    True

    """

    ###
    # Properties
    ###

    @property
    def infile(self):
        """
        Reference to the associated FCS file.

        """
        return self._infile

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
        time_channel_idx = [idx
                            for idx, channel in enumerate(self.channels)
                            if channel.lower() == 'time']
        if len(time_channel_idx) > 1:
            raise KeyError("more than one time channel in data")
        # Check if the time channel is available
        elif len(time_channel_idx) == 1:
            # Use the event list
            time_channel = self.channels[time_channel_idx[0]]
            return (self[-1, time_channel] - self[0, time_channel]) \
                * self.time_step
        elif (self._acquisition_start_time is not None and
                self._acquisition_end_time is not None):
            # Use start_time and end_time:
            dt = (self._acquisition_end_time - self._acquisition_start_time)
            return dt.total_seconds()
        else:
            return None

    @property
    def channels(self):
        """
        The name of the channels contained in `FCSData`.

        """
        return self._channels

    def amplification_type(self, channels=None):
        """
        Get the amplification type used for the specified channel(s).

        Each channel uses one of two amplification types: linear or
        logarithmic. This function returns, for each channel, a tuple of
        two numbers, in which the first number indicates the number of
        decades covered by the logarithmic amplifier, and the second
        indicates the linear value corresponding to the channel value zero.
        If the first value is zero, the amplifier used is linear

        The amplification type for channel "n" is extracted from the
        required $PnE parameter.

        Parameters
        ----------
        channels : int, str, list of int, list of str
            Channel(s) for which to get the amplification type. If None,
            return a list with the amplification type of all channels, in
            the order of ``FCSData.channels``.

        Return
        ------
        tuple, or list of tuples
            The amplification type of the specified channel(s). This is
            reported as a tuple, in which the first element indicates how
            many decades the logarithmic amplifier covers, and the second
            indicates the linear value that corresponds to a channel value
            of zero. If the first element is zero, the amplification type
            is linear.

        """
        # Check default
        if channels is None:
            channels = self._channels

        # Get numerical indices of channels
        channels = self._name_to_index(channels)

        # Get detector type of the specified channels
        if hasattr(channels, '__iter__'):
            return [self._amplification_type[ch] for ch in channels]
        else:
            return self._amplification_type[channels]

    def detector_voltage(self, channels=None):
        """
        Get the detector voltage used for the specified channel(s).

        The detector voltage for channel "n" is extracted from the $PnV
        parameter, if available.

        Parameters
        ----------
        channels : int, str, list of int, list of str
            Channel(s) for which to get the detector voltage. If None,
            return a list with the detector voltage of all channels, in the
            order of ``FCSData.channels``.

        Return
        ------
        float or list of float
            The detector voltage of the specified channel(s). If no
            information about the detector voltage is found for a channel,
            return None.

        """
        # Check default
        if channels is None:
            channels = self._channels

        # Get numerical indices of channels
        channels = self._name_to_index(channels)

        # Get detector type of the specified channels
        if hasattr(channels, '__iter__'):
            return [self._detector_voltage[ch] for ch in channels]
        else:
            return self._detector_voltage[channels]

    def amplifier_gain(self, channels=None):
        """
        Get the amplifier gain used for the specified channel(s).

        The amplifier gain for channel "n" is extracted from the $PnG
        parameter, if available.

        Parameters
        ----------
        channels : int, str, list of int, list of str
            Channel(s) for which to get the amplifier gain. If None,
            return a list with the amplifier gain of all channels, in the
            order of ``FCSData.channels``.

        Return
        ------
        float or list of float
            The amplifier gain of the specified channel(s). If no
            information about the amplifier gain is found for a channel,
            return None.

        """
        # Check default
        if channels is None:
            channels = self._channels

        # Get numerical indices of channels
        channels = self._name_to_index(channels)

        # Get detector type of the specified channels
        if hasattr(channels, '__iter__'):
            return [self._amplifier_gain[ch] for ch in channels]
        else:
            return self._amplifier_gain[channels]

    def range(self, channels=None):
        """
        Get the range of the specified channel(s).

        The range is a two-element list specifying the smallest and largest
        values that an event in a channel should have. Note that with
        floating point data, some events could have values outside the
        range in either direction due to instrument compensation.

        The range should be transformed along with the data when passed
        through a transformation function.

        The range of channel "n" is extracted from the $PnR parameter as
        ``[0, $PnR - 1]``.

        Parameters
        ----------
        channels : int, str, list of int, list of str
            Channel(s) for which to get the range. If None, return a list
            with the range of all channels, in the order of
            ``FCSData.channels``.

        Return
        ------
        array or list of arrays
            The range of the specified channel(s).

        """
        # Check default
        if channels is None:
            channels = self._channels

        # Get numerical indices of channels
        channels = self._name_to_index(channels)

        # Get the range of the specified channels
        if hasattr(channels, '__iter__'):
            return [self._range[ch] for ch in channels]
        else:
            return self._range[channels]

    def resolution(self, channels=None):
        """
        Get the resolution of the specified channel(s).

        The resolution specifies the number of different values that the
        events can take. The resolution is directly obtained from the $PnR
        parameter.

        Parameters
        ----------
        channels : int, str, list of int, list of str
            Channel(s) for which to get the resolution. If None, return a
            list with the resolution of all channels, in the order of
            ``FCSData.channels``.

        Return
        ------
        int or list of ints
            Resolution of the specified channel(s).

        """
        # Check default
        if channels is None:
            channels = self._channels

        # Get numerical indices of channels
        channels = self._name_to_index(channels)

        # Get resolution of the specified channels
        if hasattr(channels, '__iter__'):
            return [self._resolution[ch] for ch in channels]
        else:
            return self._resolution[channels]


    ###
    # Functions overriding inherited np.ndarray functions
    ###
    # For more details, see
    # http://docs.scipy.org/doc/numpy/user/basics.subclassing.html.
    ###

    # Functions involved in the creation of new arrays

    def __new__(cls, infile):

        # Load FCS file
        fcs_file = FCSFile(infile)

        ###
        # Channel-independent information
        ###

        # The time step is such that ``self[:,'Time']*time_step`` is in seconds.
        # FCS-Standard files store the time step in the $TIMESTEP keyword.
        # In CellQuest Pro's FCS2.0, the TIMETICKS keyword parameter contains
        # the time step in milliseconds.
        if '$TIMESTEP' in fcs_file.text:
            time_step = float(fcs_file.text['$TIMESTEP'])
        elif 'TIMETICKS' in fcs_file.text:
            time_step = float(fcs_file.text['TIMETICKS'])/1000.
        else:
            time_step = None

        # Data type
        data_type = fcs_file.text.get('$DATATYPE')

        # Extract the acquisition date.
        acquisition_date = cls._parse_date_string(fcs_file.text.get('$DATE'))

        # Extract the times of start and end of acquisition time.
        acquisition_start_time = cls._parse_time_string(
            fcs_file.text.get('$BTIM'))
        acquisition_end_time = cls._parse_time_string(
            fcs_file.text.get('$ETIM'))

        # If date information was available, add to acquisition_start_time and
        # acquisition_end_time.
        if acquisition_date is not None:
            if acquisition_start_time is not None:
                acquisition_start_time = datetime.datetime.combine(
                    acquisition_date,
                    acquisition_start_time)
            if acquisition_end_time is not None:
                acquisition_end_time = datetime.datetime.combine(
                    acquisition_date,
                    acquisition_end_time)

        ###
        # Channel-dependent information
        ###

        # Number of channels: Stored in the $PAR keyword parameter
        num_channels = int(fcs_file.text['$PAR'])

        # Channel names: Stored in the keyword parameter $PnN for channel n.
        channels = [fcs_file.text.get('$P{}N'.format(i))
                    for i in range(1, num_channels + 1)]
        channels = tuple(channels)

        # Amplification type: The amplification type for channel n is stored
        # in keyword parameter $PnE. This consists of a tuple of two numbers,
        # in which the first number indicates the number of decades covered by
        # the logarithmic amplifier, and the second indicates the linear value
        # corresponding to the channel value zero. If the first value is zero,
        # the amplifier used is linear. Note that it is a common non-standard
        # case to have the first value different from zero, but the second equal
        # to zero. In this case, information cannot be transformed back to the
        # linear space. The FCS3.1 standard recommends to assume that the second
        # value is one in this case.
        amplification_type = []
        for i in range(1, num_channels + 1):
            ati = fcs_file.text.get('$P{}E'.format(i))
            if ati is not None:
                # Separate by comma and convert to float
                ati = ati.split(',')
                ati = [float(atij) for atij in ati]
                # Non-standard case: if the first number is nonzero, and the
                # second is zero, convert the second value to one.
                if ati[0] != 0.0 and ati[1] == 0.0:
                    ati[1] = 1.0
                ati = tuple(ati)
            amplification_type.append(ati)
        amplification_type = tuple(amplification_type)

        # range and resolution: These are extracted from the required $PnR
        # keyword parameter. `range` is assumed to be [0, $PnR-1]. `resolution`
        # is always equal to $PnR.
        data_range = []
        resolution = []
        for ch_idx, ch in enumerate(channels):
            PnR = float(fcs_file.text.get('$P{}R'.format(ch_idx + 1)))
            data_range.append([0., PnR - 1])
            resolution.append(int(PnR))
        resolution = tuple(resolution)

        # Detector voltage: Stored in the keyword parameter $PnV for channel n.
        detector_voltage = []
        for i in range(1, num_channels + 1):
            channel_detector_voltage = fcs_file.text.get('$P{}V'.format(i))

            # The CellQuest Pro software saves the detector voltage in keyword
            # parameters BD$WORD13, BD$WORD14, BD$WORD15... for channels 1, 2,
            # 3...
            if channel_detector_voltage is None and 'CREATOR' in fcs_file.text \
                   and 'CellQuest Pro' in fcs_file.text.get('CREATOR'):
                channel_detector_voltage = fcs_file.text.get('BD$WORD{}' \
                                                             .format(12+i))

            # Attempt to cast extracted value to float
            # The FCS3.1 standard restricts $PnV to be a floating-point value
            # only. Any value that cannot be casted will be replaced with None.
            if channel_detector_voltage is not None:
                try:
                    channel_detector_voltage = float(channel_detector_voltage)
                except ValueError:
                    channel_detector_voltage = None
            detector_voltage.append(channel_detector_voltage)
        detector_voltage = tuple(detector_voltage)

        # Amplifier gain: Stored in the keyword parameter $PnG for channel n.
        amplifier_gain = []
        for i in range(1, num_channels + 1):
            channel_amp_gain = fcs_file.text.get('$P{}G'.format(i))

            # The FlowJo Collector's Edition version 7.5.110.7 software saves
            # the amplifier gain in keyword parameters CytekP01G, CytekP02G,
            # CytekP03G, ... for channels 1, 2, 3, ...
            if channel_amp_gain is None and 'CREATOR' in fcs_file.text and \
                    'FlowJoCollectorsEdition' in fcs_file.text.get('CREATOR'):
                channel_amp_gain = fcs_file.text.get('CytekP{:02d}G'.format(i))

            # Attempt to cast extracted value to float
            # The FCS3.1 standard restricts $PnG to be a floating-point value
            # only. Any value that cannot be casted will be replaced with None.
            if channel_amp_gain is not None:
                try:
                    channel_amp_gain = float(channel_amp_gain)
                except ValueError:
                    channel_amp_gain = None
            amplifier_gain.append(channel_amp_gain)
        amplifier_gain = tuple(amplifier_gain)

        # Get data from fcs_file object, and change writeable flag.
        data = fcs_file.data
        data.flags.writeable = True
        obj = data.view(cls)

        # Add FCS file attributes
        obj._infile = infile
        obj._text = fcs_file.text
        obj._analysis = fcs_file.analysis

        # Add channel-independent attributes
        obj._data_type = data_type
        obj._time_step = time_step
        obj._acquisition_start_time = acquisition_start_time
        obj._acquisition_end_time = acquisition_end_time

        # Add channel-dependent attributes
        obj._channels = channels
        obj._amplification_type = amplification_type
        obj._detector_voltage = detector_voltage
        obj._amplifier_gain = amplifier_gain
        obj._range = data_range
        obj._resolution = resolution
            
        ## this line of code used for debugging file import:
        # obj._header = fcs_file._header

        return obj

    def __array_finalize__(self, obj):
        """
        Method called after all methods of construction of the class.

        """
        # If called from explicit constructor, do nothing.
        if obj is None: return

        # Otherwise, copy attributes from "parent"
        # FCS file attributes
        self._infile = getattr(obj, '_infile', None)
        if hasattr(obj, '_text'):
            self._text = copy.deepcopy(obj._text)
        if hasattr(obj, '_analysis'):
            self._analysis = copy.deepcopy(obj._analysis)

        # Channel-independent attributes
        if hasattr(obj, '_data_type'):
            self._data_type = copy.deepcopy(obj._data_type)
        if hasattr(obj, '_time_step'):
            self._time_step = copy.deepcopy(obj._time_step)
        if hasattr(obj, '_acquisition_start_time'):
            self._acquisition_start_time = copy.deepcopy(
                obj._acquisition_start_time)
        if hasattr(obj, '_acquisition_end_time'):
            self._acquisition_end_time = copy.deepcopy(
                obj._acquisition_end_time)

        # Channel-dependent attributes
        if hasattr(obj, '_channels'):
            self._channels = copy.deepcopy(obj._channels)
        if hasattr(obj, '_amplification_type'):
            self._amplification_type = copy.deepcopy(obj._amplification_type)
        if hasattr(obj, '_detector_voltage'):
            self._detector_voltage = copy.deepcopy(obj._detector_voltage)
        if hasattr(obj, '_amplifier_gain'):
            self._amplifier_gain = copy.deepcopy(obj._amplifier_gain)
        if hasattr(obj, '_range'):
            self._range = copy.deepcopy(obj._range)
        if hasattr(obj, '_resolution'):
            self._resolution = copy.deepcopy(obj._resolution)

    # Helper functions
    @staticmethod
    def _parse_time_string(time_str):
        """
        Get a datetime.time object from a string time representation.

        The start and end of acquisition are stored in the optional keyword
        parameters $BTIM and $ETIM. The following formats are used
        according to the FCS standard:
            - FCS 2.0: 'hh:mm:ss'
            - FCS 3.0: 'hh:mm:ss[:tt]', where 'tt' is optional, and
              represents fractional seconds in 1/60ths.
            - FCS 3.1: 'hh:mm:ss[.cc]', where 'cc' is optional, and
              represents fractional seconds in 1/100ths.
        This function attempts to transform these formats to
        'hh:mm:ss:ffffff', where 'ffffff' is in microseconds, and then
        parse it using the datetime module.

        Parameters:
        -----------
        time_str : str, or None
            String representation of time, or None.

        Returns:
        --------
        t : datetime.time, or None
            Time parsed from `time_str`. If parsing was not possible,
            return None. If `time_str` is None, return None

        """
        # If input is None, return None
        if time_str is None:
            return None

        time_l = time_str.split(':')
        if len(time_l) == 3:
            # Either 'hh:mm:ss' or 'hh:mm:ss.cc'
            if '.' in time_l[2]:
                # 'hh:mm:ss.cc' format
                time_str = time_str.replace('.', ':')
            else:
                # 'hh:mm:ss' format
                time_str = time_str + ':0'
            # Attempt to parse string, return None if not possible
            try:
                t = datetime.datetime.strptime(time_str, '%H:%M:%S:%f').time()
            except:
                t = None
        elif len(time_l) == 4:
            # 'hh:mm:ss:tt' format
            time_l[3] = '{:06d}'.format(int(float(time_l[3])*1e6/60))
            time_str = ':'.join(time_l)
            # Attempt to parse string, return None if not possible
            try:
                t = datetime.datetime.strptime(time_str, '%H:%M:%S:%f').time()
            except:
                t = None
        else:
            # Unknown format
            t = None

        return t

    @staticmethod
    def _parse_date_string(date_str):
        """
        Get a datetime.date object from a string date representation.

        The FCS standard includes an optional keyword parameter $DATE in
        which the acquistion date is stored. In FCS 2.0, the date is saved
        as 'dd-mmm-yy', whereas in FCS 3.0 and 3.1 the date is saved as
        'dd-mmm-yyyy'.

        This function attempts to parse these formats, along with a couple
        of nonstandard ones, using the datetime module.

        Parameters:
        -----------
        date_str : str, or None
            String representation of date, or None.

        Returns:
        --------
        t : datetime.datetime, or None
            Date parsed from `date_str`. If parsing was not possible,
            return None. If `date_str` is None, return None

        """
        # If input is None, return None
        if date_str is None:
            return None

        # Standard format for FCS2.0
        try:
            return datetime.datetime.strptime(date_str, '%d-%b-%y')
        except ValueError:
            pass
        # Standard format for FCS3.0
        try:
            return datetime.datetime.strptime(date_str, '%d-%b-%Y')
        except ValueError:
            pass
        # Nonstandard format 1
        try:
            return datetime.datetime.strptime(date_str, '%y-%b-%d')
        except ValueError:
            pass
        # Nonstandard format 2
        try:
            return datetime.datetime.strptime(date_str, '%Y-%b-%d')
        except ValueError:
            pass

        # If none of these formats work, return None
        return None


    def _name_to_index(self, channels):
        """
        Return the channel indices for the specified channel names.

        Integers contained in `channel` are returned unmodified, if they
        are within the range of ``self.channels``.

        Parameters
        ----------
        channels : int or str or list of int or list of str
            Name(s) of the channel(s) of interest.

        Returns
        -------
        int or list of int
            Numerical index(ces) of the specified channels.

        """
        # Check if list, then run recursively
        #if hasattr(channels, '__iter__'):
        if type(channels) is list:
            return [self._name_to_index(ch) for ch in channels]
        
        if type(channels) is tuple:
            return [self._name_to_index(ch) for ch in channels]

        if isinstance(channels, str):
            # channels is a string containing a channel name
            if channels in self.channels:
                return self.channels.index(channels)
            else:
                raise ValueError("{} is not a valid channel name."
                    .format(channels))

        if isinstance(channels, int):
            if (channels < len(self.channels)
                    and channels >= -len(self.channels)):
                return channels
            else:
                raise ValueError("index out of range")

        else:
            raise TypeError("input argument should be an integer, string or "
                "list of integers or strings")

    # Functions overridden to allow string-based indexing.

    def __array_wrap__(self, out_arr, context = None):
        """
        Method called after numpy ufuncs.

        """
        if out_arr.ndim == 0:
            return out_arr[()]
        else:
            return np.ndarray.__array_wrap__(self, out_arr, context)

    def __getitem__(self, key):
        """
        Get an element or elements of the array.

        This function extends ``ndarray.__getitem__``.

        If the second value of the provided `key` is a string corresponding
        to a valid channel name, this function converts it to a number and
        passes it to ndarray's `__getitem__`. This allows for indexing by
        channel name. In addition, this function takes care of properly
        slicing the `channel_info` attribute.

        """
        # If key is a tuple with no None, decompose and interpret key[1] as 
        # the channel. If it contains Nones, pass directly to 
        # ndarray.__getitem__() and convert to np.ndarray. Otherwise, pass
        # directly to ndarray.__getitem__().
        if isinstance(key, tuple) and len(key) == 2 \
            and key[0] is not None and key[1] is not None:
            # Separate key components
            key_event = key[0]
            key_channel = key[1]

            # Convert key_channel to integers if necessary
            if not isinstance(key_channel, slice):
                key_channel = self._name_to_index(key_channel)

            # Reassemble key components
            key_all = (key_event, key_channel)

            # Get sliced array
            new_arr = np.ndarray.__getitem__(self, key_all)
            # Return if not an array
            if not hasattr(new_arr, '__iter__'):
                return new_arr

            # Finally, slice channel-dependent attributes
            if hasattr(key_channel, '__iter__'):
                new_arr._channels = tuple(
                    [new_arr._channels[kc] for kc in key_channel])
                new_arr._amplification_type = tuple(
                    [new_arr._amplification_type[kc] for kc in key_channel])
                new_arr._detector_voltage = tuple(
                    [new_arr._detector_voltage[kc] for kc in key_channel])
                new_arr._amplifier_gain = tuple(
                    [new_arr._amplifier_gain[kc] for kc in key_channel])
                new_arr._range = \
                    [new_arr._range[kc] for kc in key_channel]
                new_arr._resolution = tuple(\
                    [new_arr._resolution[kc] for kc in key_channel])
            elif isinstance(key_channel, slice):
                new_arr._channels = new_arr._channels[key_channel]
                new_arr._amplification_type = \
                    new_arr._amplification_type[key_channel]
                new_arr._detector_voltage = \
                    new_arr._detector_voltage[key_channel]
                new_arr._amplifier_gain = \
                    new_arr._amplifier_gain[key_channel]
                new_arr._range = \
                    new_arr._range[key_channel]
                new_arr._resolution = \
                    new_arr._resolution[key_channel]
            else:
                new_arr._channels = tuple([new_arr._channels[key_channel]])
                new_arr._amplification_type = \
                    tuple([new_arr._amplification_type[key_channel]])
                new_arr._detector_voltage = \
                    tuple([new_arr._detector_voltage[key_channel]])
                new_arr._amplifier_gain = \
                    tuple([new_arr._amplifier_gain[key_channel]])
                new_arr._range = \
                    [new_arr._range[key_channel]]
                new_arr._resolution = \
                    tuple([new_arr._resolution[key_channel]])

        elif isinstance(key, tuple) and len(key) == 2 \
            and (key[0] is None or key[1] is None):
            # Get sliced array and convert to np.ndarray
            new_arr = np.ndarray.__getitem__(self, key)
            new_arr = new_arr.view(np.ndarray)

        else:
            # Get sliced array using native getitem function.
            new_arr = np.ndarray.__getitem__(self, key)

        return new_arr

    def __setitem__(self, key, item):
        """
        Set an element or elements of the array.

        This function extends ``ndarray.__setitem__``.

        If the second value of the provided `key` is a string corresponding
        to a valid channel name, this function converts it to a number and
        passes it to ndarray's `__setitem__`. This allows for indexing by
        channel name when writing to a FCSData object.

        """
        # If key is a tuple with no Nones, decompose and interpret key[1] as 
        # the channel. If it contains Nones, pass directly to 
        # ndarray.__setitem__().
        if isinstance(key, tuple) and len(key) == 2 \
            and key[0] is not None and key[1] is not None:
            # Separate key components
            key_event = key[0]
            key_channel = key[1]

            # Convert key_channel to integers if necessary
            if not isinstance(key_channel, slice):
                key_channel = self._name_to_index(key_channel)

            # Reassemble key components
            key_all = (key_event, key_channel)

            # Write into array
            np.ndarray.__setitem__(self, key_all, item)

        else:
            # Get sliced array using native getitem function.
            np.ndarray.__setitem__(self, key, item)

    # Functions overridden to define printed representation.

    def __str__(self):
        """
        Return name of FCS file.

        """
        return os.path.basename(str(self.infile))


class FCSDataFrame():
    """
    Object containing events data and metadata from a flow cytometry sample.

    An `FCSDataFrame` object contains a Pandas DataFrame instance representing
    the fklow cytometry data (N cytometry events with D dimensions (channels)
    extracted from the DATA segment of an FCS file.
    Indexing of the flow cytometry data can be performed using the indexing
    methods for a Pandas DataFrame.

    It also contains a metadata object which provides an interface for our
    metadata schema - to be finalized later, for now it just has the same
    metadata that was saved as attributes of the FlowCal FCSData object.

    Information regarding the acquisition date, time, and information about
    the detector and the amplifiers are parsed from the TEXT segment of the
    FCS file and saved as part of the metadata object.

    Parameters
    ----------
    infile : str or file-like
        Reference to the associated FCS file.

    Attributes
    ----------
    flow_frame : pandas.DataFrame
        Contains the flow cytometry data

    metadata : flowgatenist.MetaData (eventually to be moved to a different
                                     package)
        Atributes of the metadata object
        --------------------------------
        infile : str or file-like
            Reference to associated FCS file.
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
    `FCSDataFrame` uses `FCSFile` to parse an FCS file. All restrictions on
    the FCS file format and the Exceptions spcecified for FCSFile also apply
    to FCSDataFrame.
    On initialization from an FCS file, the __init__ method of FCSDataFrame
    also converts all channel data to rfi using the
    flowgatenist.transform.to_rfi() method. You can always get the raw channel
    data from the FCS using the old FlowCal FCSData object, but I'm not sure
    why you would want to.

    Parsing of some non-standard files is supported [4]_.

    References
    ----------
    .. [1] P.N. Dean, C.B. Bagwell, T. Lindmo, R.F. Murphy, G.C. Salzman,
       "Data file standard for flow cytometry. Data File Standards
       Committee of the Society for Analytical Cytology," Cytometry vol
       11, pp 323-332, 1990, PMID 2340769.

    .. [2] L.C. Seamer, C.B. Bagwell, L. Barden, D. Redelman, G.C. Salzman,
       J.C. Wood, R.F. Murphy, "Proposed new data file standard for flow
       cytometry, version FCS 3.0," Cytometry vol 28, pp 118-122, 1997,
       PMID 9181300.

    .. [3] J. Spidlen, et al, "Data File Standard for Flow Cytometry,
       version FCS 3.1," Cytometry A vol 77A, pp 97-100, 2009, PMID
       19937951.

    .. [4] R. Hicks, "BD$WORD file header fields,"
       https://lists.purdue.edu/pipermail/cytometry/2001-October/020624.html

    Examples
    --------
    Load an FCS file into an FCSData object

    >>> import flowgatenist
    >>> d = flowgatenist.io.FCSDataFrame('test/Data001.fcs')

    Check channel names

    >>> print d.channels
    ('FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'Time')

    Check the size of FCSData

    >>> print d.flow_frame.shape
    (20949, 6)

    Get the first 100 events

    >>> d_sub = d.flow_frame[:100]
    >>> print d_sub.shape
    (100, 6)

    Retain only fluorescence channels

    >>> d_fl = d.flow_frame[:, ['FL1-H', 'FL2-H', 'FL3-H']]
    >>> d_fl.collumns
    Index(['FL1-H', 'FL2-H', 'FL3-H'], dtype='object')

    Channel slicing can also be done with other pandas.DataFrame slicing
    methods

    """


    # Initialization method for FCSDataFrame object:
    def __init__(self, infile):

        # Use exisitng FloweCal code to get data from FCS file and create a
        # 'FCSData' object that has the data as an ndarray, and the metadata
        # as various properties
        obj = flowgatenist.io.FCSData(infile)
            
        ## this line of code used for debugging file import:
        # self._header = obj._header

        # convert to rfi:
        obj = flowgatenist.transform.to_rfi(obj)

        self.flow_frame = pd.DataFrame(obj, columns=obj.channels)

        # Add MetaData object
        self.metadata = flowgatenist.metadata.MetaData()

        # Add FCS file attributes
        self.metadata._infile = obj.infile
        self.metadata._text = obj.text
        self.metadata._analysis = obj.analysis

        # Add channel-independent attributes
        self.metadata._data_type = obj.data_type
        self.metadata._time_step = obj.time_step
        self.metadata._acquisition_start_time = obj.acquisition_start_time
        self.metadata._acquisition_end_time = obj.acquisition_end_time

        # Change acquisition_start_time, and acquisition_end_time
        # to Pandas Timestamps
        try:
            self.metadata._acquisition_start_time = pd.to_datetime(self.metadata._acquisition_start_time)
        except ValueError:
            pass
        try:
            self.metadata._acquisition_end_time = pd.to_datetime(self.metadata._acquisition_end_time)
        except ValueError:
            pass

        # Add channel-dependent attributes
        self.metadata._channels = obj.channels
        self.metadata._amplification_type = obj.amplification_type
        self.metadata._detector_voltage = obj.detector_voltage
        self.metadata._amplifier_gain = obj.amplifier_gain
        self.metadata._range = obj.range
        self.metadata._resolution = obj.resolution

    # Functions to define printed representation.
    def __str__(self):
        """
        Return name of FCS or raw data file.

        """
        return os.path.basename(str(self.infile))
    
    def save_as_pickle(self):
        file = self.metadata._infile
        if file[-4:] == '.fcs':
            raise ValueError(f'Attempting to overwrite raw fcs file: {file}')
        
        print(f'Saving data in {file}')
        with open(file, 'wb') as f:
            pickle.dump(self, f)
