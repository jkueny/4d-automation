import glob as glob
import os
from time import sleep
import numpy as np
import subprocess
import platform

current_platform = platform.system()

# if (os.environ['HOME'].endswith('jkueny')) or (os.environ['HOME'].endswith('xsup')):
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
elif (os.environ['USERPROFILE'].endswith('PhaseCam')) and (current_platform.upper() == 'WINDOWS'):
    from fourD import *
    MessageBox('Executing on the 4D computer...')
    machine_name = '4d'
else:
    print('Unsupported platform: ', current_platform)


# from irisao import write_ptt_command, apply_ptt_command #commented 6/27/23 JKK
# from . import alpao #commented 6/27/23 JKK

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def phasecam_dm_run(
                dm_inputs,
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
                input_name='dm_input.fits',
                ispinky=True,
                ):
    '''
    Loop over dm_inputs, setting the DM in the requested state,
    and taking measurements on the 4D.

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
        dm_inputs: array-like
            Cube of displacement images. The DM will iteratively
            be set in each state on channel 0.
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

    if not (dry_run or clobber):
        # Create a new directory outname to save results to
        assert not os.path.exists(outname), '{0} already exists!'.format(outname)
        os.mkdir(outname)

    if ispinky:
        bmc1k_mon = BMC1KMonitor(localfpath,remotefpath)
    else:
        fd_mon = fourDMonitor(localfpath,remotefpath)

    for idx, inputs in enumerate(dm_inputs):

        # if dmtype.upper() == 'BMC':

        if machine_name.upper() == 'PINKY':
            #Remove any old inputs if they exist
            old_files = glob.glob(os.path.join(localfpath,'dm_input*.fits'))
            for old_file in old_files:
                if os.path.exists(old_file):
                    os.remove(old_file)
            # Write out FITS file with requested DM input
            log.info('Setting DM to state {0}/{1}.'.format(idx + 1, len(dm_inputs)))
            if not dry_run:
                dm01.write(inputs)
            input_file = os.path.join(localfpath,input_name)
            write_fits(filename=input_file, data=inputs, dtype=np.float32, overwrite=True)
            log.info('Sending new command...')
            bmc1k_mon.watch(0.1) #this is watching for new dm_inputs.fits in localfpath
        elif machine_name.upper() == '4D': #need to do this because the 4Sight
            #software can't handle fits files, outside installs not allowed...
            #so here, we need to scp a file over the network to talk to pinky
            fd_mon.watch(0.01) #this is watching for dm_ready file in localfpath
            log.info('DM ready!')
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

        # Wait until DM indicates it's in the requested state
        # I'm a little worried the DM could get there before
        # the monitor starts watching the dm_ready file, but 
        # that hasn't happened yet.
        

        if not dry_run:
            # Take an image on the Zygo
            log.info('Taking image!')
            measurement = capture_frame(reference=reference,filenameprefix=os.path.join(outname,'frame_{0:05d}.h5'.format(idx)),
                          mtype=mtype)

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

def write_dm_run_to_hdf5(filename, surface_cube, surface_attrs, intensity_cube,
                         intensity_attrs, all_attributes, dm_inputs, mask):
    '''
    Write the measured surface, intensity, attributes, and inputs
    to a single HDF5 file.

    Attempting to write out the Mx dataset attributes (surface, intensity)
    currently breaks things (Python crashes), so I've disabled that for now.
    All the information *should* be in the attributes group, but it's
    not as convenient.

    Parameters:
        filename: str
            File to write out consolidate data to
        surface_cube : nd array
            Cube of surface images
         surface_attrs : dict or h5py attributes object
            Currently not used, but expected.
         intensity_cube : nd array
            Cube of intensity images
        intensity_attrs : dict or h5py attributes object
            Currently not used, but expected
        all_attributes : dict or h5py attributes object
            Mx attributes to associate with the file.
        dm_inputs : nd array
            Cube of inputs for the DM
        mask : nd array
            2D mask image
    Returns: nothing
    '''
    import h5py

    # create hdf5 file
    f = h5py.File(filename, 'w')
    
    # surface data and attributes
    surf = f.create_dataset('surface', data=surface_cube)
    #surf.attrs.update(surface_attrs)

    intensity = f.create_dataset('intensity', data=intensity_cube)
    #intensity.attrs.update(intensity_attrs)

    attributes = f.create_group('attributes')
    attributes.attrs.update(all_attributes)

    dm_inputs = f.create_dataset('dm_inputs', data=dm_inputs)
    #dm_inputs.attrs['units'] = 'microns'

    mask = f.create_dataset('mask', data=mask)
    
    f.close()

def update_status_file(localfpath,remotefpath,user,address):
    '''
    Write an empty file at the correct folder, given the machine
    '''
    send_to = '{}@{}:{}'.format(user,address,remotefpath) 
    try:
        print('Attempting to send to {}'.format(send_to))
        subprocess.run(['scp', localfpath, send_to], check=True)
        print('File copied to {}'.format(send_to))
    except subprocess.CalledProcessError as e:
        print('Error: {}'.format(e))



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
        to_address = '192.168.1.2'

        # Write out empty file locally, then scp over to tell 4Sight the DM is ready.
        open(local_status_fname, 'w').close()
        update_status_file(localfpath=local_status_fname,
                           remotefpath=self.remote_send,
                           user=to_user,address=to_address)


class BMC1KMonitor(FileMonitor):
    '''
    Set the DM machine to watch a particular FITS files for
    a modification, indicating a request for a new DM actuation
    state.

    Will ignore the current file if it already exists
    when the monitor starts (until it's modified).
    '''
    def __init__(self, locpath, rempath, input_file='dm_input.fits'):
        '''
        Parameters:
            locpath : str
                Local path to create and watch for status files.
            rempath: str
                Remote path to scp status files to.
        '''
        self.remote_send = rempath
        super().__init__(os.path.join(locpath, input_file))

    def on_new_data(self, newdata):
        '''
        On detecting an updated dm_input.fits file,
        load the image onto the DM and scp an
        empty 'dm_ready' file to the 4D machine
        '''
        # Load image from FITS file onto DM channel 0
        log.info('Setting DM from new image file {}'.format(newdata))
        to_user = 'PhaseCam'
        to_address = '192.168.1.3'
        # load_channel(newdata, 1) #dmdisp01
        local_status_fname = os.path.join(os.path.dirname(self.file), 'dm_ready')

        # Write out empty file locally, then scp over to tell 4Sight the DM is ready.
        open(local_status_fname, 'w').close()
        update_status_file(localfpath=local_status_fname,
                           remotefpath=self.remote_send,
                           user=to_user,address=to_address)



if __name__ == '__main__':
    kilo_dm_size = (34,34)
    n_actuators = 952
    voltage_bias = -1
    save_measure_dir = "C:\\Users\\PhaseCam\\Documents\\jay_4d\\4d-automation\\test"
    reference_flat = "C:\\Users\\PhaseCam\\Documents\\jay_4d\\reference_lamb20avg12_average_ttp-removed.h5"
    if machine_name.upper() == 'PINKY':
        home_folder = "/home/jkueny"
        remote_folder = 'C:\\Users\\PhaseCam'
    elif machine_name.upper() == 'PHASECAM':
        home_folder = 'C:\\Users\\PhaseCam'
        remote_folder = "/home/jkueny"
    else:
        print('Error, what machine? Bc apparently it is not pinky or the 4D machine...')
    # kilo_map = np.load('/opt/MagAOX/calib/dm/bmc_1k/bmc_2k_actuator_mapping.npy')
    kilo_map = np.load('/opt/MagAOX/calib/dm/bmc_1k/bmc_2k_actuator_mapping.npy')
    kilo_mask = (kilo_map > 0)
    cmds_matrix = voltage_bias * np.eye(kilo_dm_size[0]*kilo_dm_size[1])[kilo_mask.flatten()]
    dm_cmds = cmds_matrix.reshape(n_actuators,kilo_dm_size[0],kilo_dm_size[1])
    single_pokes = []
    for i in range(len(dm_cmds[0])): #34 length
        single_pokes.append(dm_cmds[i])
        break #starting with one command for now
    print(len(single_pokes))
    phasecam_dm_run(dm_inputs=single_pokes,
                    localfpath=home_folder,
                    remotefpath=remote_folder,
                    outname=save_measure_dir,
                    reference=reference_flat,
                    dry_run=True,)

# class ZygoMonitor(FileMonitor):
#     '''
#     Set the Zygo machine to watch for an indication from
#     the DM that it's been put in the requested state,
#     and proceed with data collection when ready
#     '''
#     def __init__(self, path):
#         '''
#         Parameters:
#             path : str
#                 Network path to watch for 'dm_ready'
#                 file indicating the DM is in the
#                 requested state.
#         '''
#         super().__init__(os.path.join(path,'dm_ready'))

#     def on_new_data(self, newdata):
#         '''
#         On detecting a new 'dm_ready' file,
#         stop blocking the Zygo code. (No 
#         actual image capture happens here.)
#         '''
#         os.remove(newdata) # delete DM ready file
#         self.continue_monitoring = False # stop monitor loop

# class BMC2KMonitor(FileMonitor):
#     '''
#     Set the DM machine to watch a particular FITS files for
#     a modification, indicating a request for a new DM actuation
#     state.

#     Will ignore the current file if it already exists
#     when the monitor starts (until it's modified).
#     '''
#     def __init__(self, path, serial, input_file='dm_input.fits', script_path='/home/kvangorkom/BMC-interface'):
#         '''
#         Parameters:
#             path : str
#                 Network path to watch for 'dm_input.fits'
#                 file.
#         '''
#         super().__init__(os.path.join(path, input_file))
#         self.serial = serial
#         self.script_path = script_path

#     def on_new_data(self, newdata):
#         '''
#         On detecting an updated dm_input.fits file,
#         load the image onto the DM and write out an
#         empty 'dm_ready' file to the network path
#         '''
#         # Load image from FITS file onto DM channel 0
#         log.info('Setting DM from new image file {}'.format(newdata))
#         update_voltage_2K(newdata, self.serial, self.script_path)

#         # Write out empty file to tell Zygo the DM is ready.
#         open(os.path.join(os.path.dirname(self.file), 'dm_ready'), 'w').close()

# class ALPAOMonitor(FileMonitor):
#     '''
#     Set the DM machine to watch a particular FITS files for
#     a modification, indicating a request for a new DM actuation
#     state.

#     Will ignore the current file if it already exists
#     when the monitor starts (until it's modified).
#     '''
#     def __init__(self, path, serial, input_file='dm_input.fits'):
#         '''
#         Parameters:
#             path : str
#                 Network path to watch for 'dm_input.fits'
#                 file.
#             serial : str
#                 ALPAO DM97 serial number. Probably "BAX150"
#         '''
#         super().__init__(os.path.join(path, input_file))
#         self.serial = serial
#         #self.img = alpao.link_to_shmimage(serial)

#     def on_new_data(self, newdata):
#         '''
#         On detecting an updated dm_input.fits file,
#         load the image onto the DM and write out an
#         empty 'dm_ready' file to the network path
#         '''
#         # Load image from FITS file onto DM channel 0
#         log.info('Setting DM from new image file {}'.format(newdata))
#         #alpao.apply_command(fits.open(newdata)[0].data, self.serial, self.img)
#         alpao.apply_command_from_fits(newdata, self.serial)

#         # Write out empty file to tell Zygo the DM is ready.
#         open(os.path.join(os.path.dirname(self.file), 'dm_ready'), 'w').close()

# class IrisAOMonitor(FileMonitor):
#     '''
#     Set the DM machine to watch a particular FITS files for
#     a modification, indicating a request for a new DM actuation
#     state.

#     Will ignore the current file if it already exists
#     when the monitor starts (until it's modified).
#     '''
#     def __init__(self, path, mserial, input_file='ptt_input.txt'):
#         '''
#         Parameters:
#             path : str
#                 Network path to watch for 'ptt_input.txt'
#                 file.
#         '''
#         super().__init__(os.path.join(path, input_file))
#         self.mserial = mserial

#     def on_new_data(self, newdata):
#         '''
#         On detecting an updated dm_input.fits file,
#         load the image onto the DM and write out an
#         empty 'dm_ready' file to the network path
#         '''
#         # Load image from FITS file onto DM channel 0
#         log.info('Setting DM from new PTT file {}'.format(newdata))
#         apply_ptt_command(newdata, mserial=self.mserial)

#         # Write out empty file to tell Zygo the DM is ready.
#         open(os.path.join(os.path.dirname(self.file), 'dm_ready'), 'w').close()

# class BaslerMonitor(FileMonitor):
#     def __init__(self, path, camera, images, stop_after_capture=False, nimages=1):
#         '''
#         Parameters:
#             path : str
#                 Network path to watch for 'dm_ready'
#                 file indicating the DM is in the
#                 requested state.
#             camera : pypylon camera object
#             images : list
#                 List to append images to
#             stop_after_capture : bool, opt.
#                 Stop monitor after capturing an 
#                 image? Default: False.
#             nimages : int, opt.
#                 Take multiple images? If > 1, each element
#                 of the image list will be an array of images
#         '''
#         super().__init__(os.path.join(path,'dm_ready'))
        
#         self.camera = camera
#         self.images = images
#         self.stop_after_capture = stop_after_capture
#         self.nimages = nimages

#     def on_new_data(self, newdata):
#         '''
#         On detecting a new 'dm_ready' file, capture
#         an image on the Basler camera.
#         '''
#         if self.nimages == 1:
#             self.images.append(self.camera.grab_image().astype(float))
#         else:
#             self.images.append(np.asarray(list(self.camera.grab_images(self.nimages))).astype(float))
#         log.info('Grabbed Basler frame! ({})'.format(len(self.images)))
#         open(os.path.join(os.path.dirname(self.file), 'basler_ready'), 'w').close()
#         if self.stop_after_capture:
#             self.continue_monitoring = False
