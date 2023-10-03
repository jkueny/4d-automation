import glob as glob
import os
from time import sleep
import numpy as np
import subprocess
import platform

current_platform = platform.system()

# if (os.environ['HOME'].endswith('jkueny')) or (os.environ['HOME'].endswith('xsup')):
try:
    if (os.environ['HOME'].endswith('jkueny')) and (current_platform.upper() == 'LINUX'):
        print('Starting 4D automation script...')
        # import h5py
        from astropy.io import fits
        from bmc import load_channel, write_fits, update_voltage_2K
        from magpyx.utils import ImageStream
        # from magpyx.dm import dmutils
        print('Executing on Pinky...')
        machine_name = 'pinky'
        dm01 = ImageStream('dm01disp01')
except:
    if (os.environ['USERPROFILE'].endswith('PhaseCam')) and (current_platform.upper() == 'WINDOWS'):
        from fourD import *
        MessageBox('Executing on the 4D computer...')
        machine_name = '4d'
    else:
        print('Unsupported platform: ', current_platform)

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def phasecam_run(
                localfpath,
                remotefpath,
                outname,
                # dmtype,
                delay=None,
                # consolidate=True,
                dry_run=True,
                clobber=False,
                reference=None,
                mtype='average',
                input_name='dm_ready',
                ):
    '''
    Ultimately we probably want to perform a "measurement" b/w
    DM commands which produces Surface Data. The Raw Data is
    just the unwrapped phase map, while the Surface Data has the
    reference subtracted, Zernike removal, and masking applied.

    In the outname directory, the individual measurements are
    saved in separate .hdf5 files. 
    
    Consolidated measurements (surface maps, intensity maps,
    attributes, dm inputs) are saved out to 'alldata.hdf5' under
    this directory. (Not really done here b/c 4Sight outputs hdf5)
    by default. At the end, we can probably use the GUI to consolidate
    all .hdf5's if that's what works best. JKK 06/28/23)

    Parameters:
        localfpath : str
            Path to folder where cross-machine communication 
            will take place. Both machines must have read/write 
            privileges.
        dmtype : str
            'bmc', 'irisao', or 'alpao'. This determines whether
            the dm_inputs are written to .fits or .txt. Disabled,
            because this is only for our new Kilo from BMC for now.
            JKK 09/19/23
        outname : str
            Directory to write results out to. Directory
            must not already exist.
        delay : float, opt.
            Time in seconds to wait between measurements.
            Default: no delay.
        consolidate : bool, opt.
            Attempt to consolidate all files at the end?
            Default: True
        dry_run : bool, opt.
            If toggled to True, this will loop over DM states
            without taking images. This is useful to debugging
            things on the DM side / watching the fringes on
            the Zygo live monitor.
        clobber : bool, opt.
            Allow writing to directory that already exists?
            Risks overwriting files that already exist, but
            useful for interactive measurements.
        mtype : str
            'acquire' or 'measure'. 'Acquire' takes a measurement
            without analyzing or updating the GUI (faster), while
            'measure' takes a measurement, analyzes, and updates
            the GUI (slower). JKK: Measurement is by default a 
            10 frame acquisiton, averaged to help with bench
            seeing.
    Returns: nothing

    '''
    # if dmtype.upper() not in ['BMC']:
    #     raise ValueError('dmtype not recognized. Must be "BMC".')

    input_file = os.path.join(localfpath,input_name)
    if not (dry_run or clobber):
        # Create a new directory outname to save results to
        assert not os.path.exists(outname), '{0} already exists!'.format(outname)
        assert not os.path.exists(input_file), '{0} aready exists! Aborting...'.format(outname)
        os.mkdir(outname)

    fd_mon = fourDMonitor(localfpath,remotefpath)

    for i in range(3): #iterations
        #software can't handle fits files, outside installs not allowed...
        #so here, we need to scp a file over the network to talk to pinky
        fd_mon.watch(0.01) #this is watching for dm_ready file in localfpath
        log.info('DM ready!')
        # Wait until DM indicates it's in the requested state
        # I'm a little worried the DM could get there before
        # the monitor starts watching the dm_ready file, but 
        # that hasn't happened yet.
        

        if not dry_run:
            # Take an image on the Zygo
            log.info('Taking image!')
            measurement = capture_frame(reference=reference,
                                        filenameprefix=os.path.join(outname,'frame_{0:05d}.h5'.format(i)),
                                        mtype=mtype)
        else:
            raise NameError('Which machine is executing this script?') 
        # write out
        # if dmtype.upper() == 'ALPAO':
        #     alpao.command_to_fits(inputs, input_file, overwrite=True)
        # else: #BMC
        #     write_fits(input_file, inputs, dtype=np.float32, overwrite=True)
        # else: #IRISAO
        #     log.info('Invalid DM type. Only BMC is currently supported for PhaseCam work.')
        #     break
        # else: #IRISAO
        #     input_file = os.path.join(localfpath,'ptt_input.txt'.format(idx))
        #     write_ptt_command(inputs, input_file)


        # Remove input file
        if os.path.exists(input_file):
            os.remove(input_file)

        if delay is not None:
            sleep(delay)

    # if consolidate:
    #     log.info('Writing to consolidated .hdf5 file.')
    #     # Consolidate individual frames and inputs
    #     # Don't read attributes into a dictionary. This causes python to crash (on Windows)
    #     # when re-assignging them to hdf5 attributes.
    #     alldata = read_many_raw_datx(sorted(glob.glob(os.path.join(outname,'frame_*.datx'))), 
    #                                  attrs_to_dict=True, mask_and_scale=True)
    #     write_dm_run_to_hdf5(os.path.join(outname,'alldata.hdf5'),
    #                          np.asarray(alldata['surface']),
    #                          alldata['surface_attrs'][0],
    #                          np.asarray(alldata['intensity']),
    #                          alldata['intensity_attrs'][0],
    #                          alldata['attrs'][0],
    #                          np.asarray(dm_inputs),
    #                          alldata['mask'][0]
    #                          )

class FileMonitor(object):
    '''
    Watch a file for modifications at some
    cadence and perform some action when
    it's modified.  
    '''
    def __init__(self, file_to_watch):
        '''
        Parameters:
            file_to_watch : str
                Full path to a file to watch for.
                On detecting a modificiation, do
                something (self.on_new_data)
        '''
        self.file = file_to_watch
        self.continue_monitoring = True

        # Find initial state
        self.last_modified = self.get_last_modified(self.file)

    def watch(self, period=1.):
        '''
        Pick out new data that have appeared since last query.
        Period given in seconds.
        '''
        self.continue_monitoring = True
        try:
            while self.continue_monitoring:
                # Check the file
                last_modified = self.get_last_modified(self.file)

                # If it's been modified (and not deleted) perform
                # some action and update the last-modified time.
                if last_modified != self.last_modified:
                    if os.path.exists(self.file):
                        self.on_new_data(self.file)
                    self.last_modified = last_modified

                # Sleep for a bit
                sleep(period)
        except KeyboardInterrupt:
            return

    def get_last_modified(self, file):
        '''
        If the file already exists, get its last
        modified time. Otherwise, set it to 0.
        '''
        if os.path.exists(file):
            last_modified = os.stat(file).st_mtime
        else:
            last_modified = 0.
        return last_modified

    def on_new_data(self, newdata):
        ''' Placeholder '''
        pass

class fourDMonitor(FileMonitor):
    '''
    Set the 4D machine to watch for an indication from
    the DM that it's been put in the requested state,
    and proceed with data collection when ready
    '''
    def __init__(self, locpath, rempath):
        '''
        Parameters:
            locpath : str
                Local path to watch for 'dm_ready'
                file indicating the DM is in the
                requested state.
            rempath: str
                Remote path to scp status file to.
        '''
        self.remote_send = rempath
        super().__init__(os.path.join(locpath,'dm_ready'))

    def on_new_data(self, newdata):
        '''
        On detecting a new 'dm_ready' file,
        stop blocking the 4D code. (No 
        actual image capture happens here.)
        '''
        os.remove(newdata) # delete DM ready file
        self.continue_monitoring = False # stop monitor loop
        local_status_fname = os.path.join(os.path.dirname(self.file), 'awaiting_dm')

        to_user = 'jkueny'
        to_address = '192.168.1.6'

        # Write out empty file locally, then scp over to tell 4Sight the DM is ready.
        open(local_status_fname, 'w').close()
        update_status_file(localfpath=local_status_fname,
                           remotefpath=self.remote_send,
                           user=to_user,address=to_address)
        
if __name__ == '__main__':
    save_measure_dir = "C:\\Users\\PhaseCam\\Documents\\jay_4d\\4d-automation\\test"
    #first take a flat, then hard-code the fpath for it here
    reference_flat = "C:\\Users\\PhaseCam\\Documents\\jay_4d\\reference_lamb20avg12_average_ttp-removed.h5"
    if machine_name.upper() == 'PINKY':
        print('Execution on the wrong computer!!!')
        print('We are on {}'.format(current_platform))
        print('We should be on the 4D Windows machine in Lab 584...')
    elif machine_name.upper() == 'PHASECAM':
        home_folder = 'C:\\Users\\PhaseCam'
        remote_folder = "/home/jkueny"
    else:
        print('Error, what machine? Bc apparently it is not pinky or the 4D machine...')
    # kilo_map = np.load('/opt/MagAOX/calib/dm/bmc_1k/bmc_2k_actuator_mapping.npy')
    phasecam_run(
                    localfpath=home_folder,
                    remotefpath=remote_folder,
                    outname=save_measure_dir,
                    reference=reference_flat,
                    dry_run=True,)