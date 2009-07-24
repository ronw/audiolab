# Some of this is based on Doug Eck's code:
# http://www.iro.umontreal.ca/~eckdoug/python_for_courses/readmp3.py

import numpy as np
import mad

import pysndfile
import MP3Info

def SndfileFactory(*args, **kwargs):
    """Returns the appropriate Sndfile object based on the format of
    the given file.
    """
    try:
        sndfile = MP3Sndfile(*args, **kwargs)
    except:
        sndfile = pysndfile.Sndfile(*args, **kwargs)
    return sndfile

class MP3Sndfile(object):
    """Implementation of Sndfile API for MP3 files based on pymad."""

    def __init__(self, filename, mode='r', format=None, channels=0,
                 samplerate=0):
        if mode != 'r':
            raise (ValueError,
                   'Only mode=''r'' (reading MP3 files) is allowed')
        self.mode = 'r'

        with open(filename) as f:
            try:
                MP3Info.MPEG(f)
            except:
                raise (IOError,
                       '%s does not appear to be an MP3 file' % filename)

        mf = mad.MadFile(filename)
        self._filename = filename
        self._madfile = mf
        # MAD can only read or seek to times aligned with MP3 frames,
        # so we need to store a local buffer of (at most) one frame to
        # allow for seeking to and reading from arbitrary samples in
        # the file.
        self._read_buffer = np.empty(0, np.int16) 
        # Frame number of first frame of self._read_buffer
        self._read_buffer_frame_index = 0
       
        self.samplerate = mf.samplerate()
        # pymad speaks in milliseconds
        self.nframes = mf.total_time() * mf.samplerate() / 1000

        mode_to_channel = {mad.MODE_SINGLE_CHANNEL:1, 
                           mad.MODE_DUAL_CHANNEL:2,
                           mad.MODE_JOINT_STEREO:2,
                           mad.MODE_STEREO:2}
        self.channels = mode_to_channel[mf.mode()] 
        self.encoding = 'mp3'
       
    def close(self):
        self._madfile = None

    def sync(self):
        pass

    def write_frames(self, frames):
        raise (NotImplementedError,
               '%s does not support writing to MP3 files' % self.__name__)

    def seek(self, offset, whence=0, mode='r'):
        """Seek to the given point

        Warning: this is really slow
        """
        if not self._madfile:
            raise IOError, 'Cannot seek on closed file'

        if whence == 0:
            pass
        elif whence == 1:
            offset = self._read_buffer_frame_index + offset
        elif whence == 2:
            offset = self.nframes + offset
        else:
            raise ValueError, 'whence must be 0, 1, or 2'

        if offset < 0 or offset > self.nframes:
            raise IOError, 'Cannot seek past end of file'

        self.close()
        self.__init__(self._filename)
        #self.read_frames(offset)
        #
        # It is a bit faster to break this into multiple calls to
        # self.read_frame to avoid excessive concatenation of numpy
        # arrays in _read_next_mp3_frame_into_buffer()
        nread = 0
        bufsize = 11520
        while nread < offset:
            if offset - nread < bufsize:
                bufsize = offset - nread
            self.read_frames(bufsize)
            nread += bufsize
        return offset

    # FIXME - this does not work (at least partially because
    # MadFile.seek_time() does not work -- check the source!)
    def __broken_seek(self, offset, whence=0, mode='r'):
        if not self._madfile:
            raise IOError, 'Cannot seek on closed file'

        if whence == 0:
            pass
        elif whence == 1:
            offset = self._read_buffer_frame_index + offset
        elif whence == 2:
            offset = self.nframes + offset
        else:
            raise ValueError, 'whence must be 0, 1, or 2'

        buflen = len(self._read_buffer) * self.channels
        if( self._read_buffer_frame_index
            <= offset < self._read_buffer_frame_index + buflen):
            start = self._read_buffer_frame_index + buflen - offset
            self._read_buffer = self._read_buffer[start:]
        else:
            # Either offset < self._read_buffer_frame_index
            # or offset > self._read_buffer_frame_index + buflen
            
            # Need to seek with pymad and store any leftover stuff in
            # self._read_buffer.

            offset_in_msec = int(offset * 1000.0 / self.samplerate) 
            self._madfile.seek_time(offset_in_msec)
            nremaining = int(offset - offset_in_msec * self.samplerate / 1000.0)

            nremaining_samples = nremaining * self.channels
            while len(self._read_buffer) < nremaining_samples:
                self._read_next_mp3_frame_into_buffer()
            self._read_buffer = self._read_buffer[nremaining_samples:]
        self._read_buffer_frame_index = offset
        return offset

    def _read_next_mp3_frame_into_buffer(self, nframes=0):
        buf = self._madfile.read()
        if buf is None:
            raise (RuntimeError, 'Asked %d frames, read %d'
                   % (nframes, len(self._read_buffer)))
        # pymad appears to only output 16 bit PCM
        tmp = np.fromstring(buf, np.int16)
        self._read_buffer = np.concatenate((self._read_buffer, tmp))

    def read_frames(self, nframes, dtype=np.float32):
        if not self._madfile:
            raise(IOError, 'Cannot read from closed file')
        if nframes < 0:
            raise(ValueError,
                  'Number of frames has to be >= 0 (was %d)' % nframes)

        # pymad returns stereo data with identical left and right
        # channels if the file is mono.
        if self.channels == 1:
            nsamples = nframes * 2
        else:
            nsamples = nframes * self.channels

        while len(self._read_buffer) < nsamples:
            self._read_next_mp3_frame_into_buffer(nframes)

        frames = self._read_buffer[:nsamples]
        if self.channels > 1:
            frames = frames.reshape((nframes, self.channels))
        self._read_buffer = self._read_buffer[nsamples:]
        self._read_buffer_frame_index += nframes

        if self.channels == 1:
            frames = frames[::2]
            frames.shape = (len(frames),)

        frames = np.array(frames, dtype=dtype)
        # Follows libsndfile conversion rules from int16 to float samples.
        # See http://www.mega-nerd.com/libsndfile/api.html#note1
        if np.issubdtype(dtype, np.dtype('float')):
            frames = frames * 1.0 / 2**15

        return frames


def mp3read(filename, last=None, first=0):
    """
    Simple reader for MP3 audio files.
 	
    Parameters
    ----------
    filename : str
        Name of the file to read
    last : int
        Last frame to read. If None, this is equal to the number of frames in
	    the file.
    first : int
        First frame to read. If 0, means starting from the beginning of the
        file.
	
    Returns
    -------
    data : array
        the read data (one column per channel)
    fs : int
        the sampling rate
    enc : str
        the encoding of the file, such as 'pcm16', 'float32', etc...

    Notes
    -----
    For a total control over options, such as output's dtype, etc...,
    you should use MP3Sndfile class instances instead
    """

    hdl = MP3Sndfile(filename, 'r')
    try:
        fs = hdl.samplerate
        enc = hdl.encoding
        # Set the pointer to start position
        nf  = hdl.seek(first, 1)
        if not nf == first:
            raise IOError("Error while seeking at starting position")
	
        if last is None:
            last = hdl.nframes
        data = hdl.read_frames(last - first)
    finally:
        hdl.close()
        
    return data, fs, enc
