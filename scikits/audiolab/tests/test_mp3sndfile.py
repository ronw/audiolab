#!/usr/bin/env python
"""Tests for the MP3Sndfile class"""

import random
import os

import numpy as np
from numpy.testing import *

from scikits.audiolab import MP3Sndfile, mp3read, wavread

from testcommon import TEST_DATA_DIR
#TEST_DATA_DIR = '/home/ronw/tmp'

class TestMP3Sndfile():
    def test_MP3Sndfile_constructor(self): 
        random.seed(1)
        sndfile = MP3Sndfile(self.mp3filename)
        self.assertEqual(sndfile.samplerate, 44100)

        self.assertRaises(IOError, MP3Sndfile, self.wavfilename)

    def test_read_frames_basic_functionality(self):
        sndfile = MP3Sndfile(self.mp3filename)

        for bufsize in [1, 10, 100, 1000, sndfile.nframes / 2]:
            buf = sndfile.read_frames(bufsize)
            self.assertEqual(len(buf), bufsize)

        self.assertRaises(ValueError, sndfile.read_frames, -10)
        self.assertRaises(RuntimeError, sndfile.read_frames, sndfile.nframes)

        # Should not be able to read the file after it is closed.
        sndfile.close()
        self.assertRaises(IOError, sndfile.read_frames, 1)

    def test_read_frames_dtype_conversion(self):
        sndfile = MP3Sndfile(self.mp3filename)
        # Number of audio frames (1 sample per channel) in an MP3 frame.
        bufsize = 1152
        dtypes = ['int8', 'int16', 'int32', 'float32', 'float64']
        for dtype in dtypes:
            buf = sndfile.read_frames(bufsize, dtype=dtype)
            self.assert_(buf.dtype is np.dtype(dtype))
        sndfile.close()

        sndfile = MP3Sndfile(self.mp3filename)
        buf = sndfile.read_frames(sndfile.nframes, dtype=np.int16)
        self.assert_(buf.max() <= 2**15)
        self.assert_(buf.min() >= -2**15)
        sndfile.close()

        # Floating point dtypes should be normalized
        sndfile = MP3Sndfile(self.mp3filename)
        buf = sndfile.read_frames(sndfile.nframes, dtype=np.float32)
        self.assert_(buf.max() <= 1.0)
        self.assert_(buf.min() >= -1.0)
        sndfile.close()

    def test_consecutive_calls_to_read_frames_do_not_lose_samples(self):
        sndfile = MP3Sndfile(self.mp3filename)
        buf = sndfile.read_frames(1)
        for incr in [250, 500, 1000, 2000, 4000, 8000]:
            buf = np.concatenate((buf, sndfile.read_frames(incr)))
        sndfile.close()

        sndfile = MP3Sndfile(self.mp3filename)
        buf2 = sndfile.read_frames(len(buf))
        sndfile.close()
        
        assert_array_equal(buf, buf2)

    def test_seek(self):
        # Based on TestSeek class from audiolab's test_sndfile.py.

        def _seek_test(filename, bufsize):
            sndfile = MP3Sndfile(filename)
            nframes = sndfile.nframes
            buf = sndfile.read_frames(bufsize)
            sndfile.seek(0)
            buf2 = sndfile.read_frames(bufsize)
            assert_array_equal(buf, buf2)
            sndfile.close()

            # Now, read some frames, go back, and compare buffers
            # (check whence == 1 == SEEK_CUR)
            sndfile = MP3Sndfile(filename)
            sndfile.read_frames(bufsize)
            buf = sndfile.read_frames(bufsize)
            starttime = sndfile._madfile.current_time()
            sndfile.seek(-bufsize, 1)
            buf2 = sndfile.read_frames(bufsize)
            assert_array_equal(buf, buf2)
            sndfile.close()

            # Now, read some frames, go back, and compare buffers
            # (check whence == 2 == SEEK_END)
            sndfile = MP3Sndfile(filename)
            buf = sndfile.read_frames(nframes)
            sndfile.seek(-bufsize, 2)
            buf2 = sndfile.read_frames(bufsize)
            assert_array_equal(buf[-bufsize:], buf2)
            sndfile.close()

        # Test seeking distances both smaller and larger than an MP3 frame.
        for bufsize in [10, 500, 5000, 10000]:
            _seek_test(self.mp3filename, bufsize)
            
        # Try to seek past the end.
        sndfile = MP3Sndfile(self.mp3filename)
        self.assertRaises(IOError, sndfile.seek, sndfile.nframes + 1)
        sndfile.close()

    def test_mp3read(self):
        sndfile = MP3Sndfile(self.mp3filename)

        d, fs, enc = mp3read(self.mp3filename)
        self.assertEqual(len(d), sndfile.nframes)
        self.assertEqual(fs, sndfile.samplerate)

        first = 0
        last = fs
        d2, fs, enc = mp3read(self.mp3filename, last)
        assert_array_equal(d[:last], d2)
        d2, fs, enc = mp3read(self.mp3filename, first=first)
        assert_array_equal(d[first:], d2)
        d2, fs, enc = mp3read(self.mp3filename, last, first)
        assert_array_equal(d[first:last], d2)

    def test_mp3read_compatibility_with_wavread(self):
        dmp3, fsmp3, encmp3 = mp3read(self.mp3filename)
        dwav, fswav, encwav = wavread(self.wavfilename)

        self.assertEquals(fsmp3, fswav)

        # Compensate for delay in mp3 file.
        #corr = np.array([sum(dmp3[n:] * dwav[:len(dmp3)-n])
        #                    for n in xrange(2000)])
        #mp3_delay = corr.argmax()
        #self.assertEqual(mp3_delay, 1105)
        mp3_delay = 1105;
        dmp3 = dmp3[mp3_delay:]
        dwav = dwav[:len(dmp3)]
        rmse = np.sqrt(np.mean((dwav - dmp3)**2))
        self.assert_(rmse < 1e-4)

class TestMP3Sndfile_Mono(TestMP3Sndfile, NumpyTestCase):
    mp3filename = os.path.join(TEST_DATA_DIR, 'mp3test.mp3')
    wavfilename = os.path.join(TEST_DATA_DIR, 'mp3test.wav')

class TestMP3Sndfile_Stereo(TestMP3Sndfile, NumpyTestCase):
    mp3filename = os.path.join(TEST_DATA_DIR, 'mp3test_stereo.mp3')
    wavfilename = os.path.join(TEST_DATA_DIR, 'mp3test_stereo.wav')
        
        
if __name__ == '__main__':
    #NumpyTest('mp3sndfile').testall()
    import unittest
    unittest.main()
    pass
