import glob as glob
import os
from time import sleep
import numpy as np
# import subprocess
import platform
# from zernike import RZern

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
        #maybe we just add numpy arrays and write to a single disp thing
        # dm02 = ImageStream('dm01disp02') #TODO need to verify this, and add an additional?
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

def dm_run(
            dmglobalbias,
            networkpath,
            coeffsfname,
            startover=True,
            niterations=3,
            delay=None,
            # consolidate=True,
            dry_run=True,
            clobber=False,
            reference=None,
            pupildiam=34,
            input_name='dm_ready',
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

    # if not (dry_run or clobber):
    #     # Create a new directory outname to save results to
    #     assert not os.path.exists(outname), '{0} already exists!'.format(outname)
    #     os.mkdir(outname)

    #Generate the Zernike polynomials now, to avoid doing it in a loop.
    #Because the 4D uses a weird ordering scheme, hard-coding this indexing
    #for now. Starting with the first 16 polynomials in Zemax(?) ordering.
    nm_pairs = [(0,0),(1,1),(1,-1),(2,0),(2,2),(2,-2),(3,1),(3,-1),
            (4,0),(3,3),(3,-3),(4,2),(4,-2),(5,1),(5,-1),(6,0)]
    zernike_polys = generate_zernike_images(kilo_dm_width, nm_pairs)
    print('Initiating FileMonitor...')
    bmc1k_mon = BMC1KMonitor(networkpath)
    #start from fresh, this zero array added to the optimal bias command.
    for i in range(niterations): #just do a few iterations for now
        if startover:
            surface_fit = np.zeros_like(dmglobalbias)
        #otherwise load the coefficient array in the shared network folder and apply
        #it to the DM as the first command.
        else:
            previous_coeffs = np.load(os.path.join(networkpath,coeffsfname)) #array of Z coeffs
            surface_fit = coeffs_to_command(previous_coeffs,zernike_polys)
        log.info('Sending new command...')
        total_command = dmglobalbias - surface_fit
        #I don't think we will save the commands as files for this job, commenting below.
        # #Remove any old inputs if they exist
        # old_files = glob.glob(os.path.join(networkpath,'dm_input*.fits'))
        # for old_file in old_files:
        #     if os.path.exists(old_file):
        #         os.remove(old_file)
        # Write out FITS file with requested DM input
        log.info('Setting DM to state {0}/{1}.'.format(i + 1, niterations))
        if not dry_run:
            dm01.write(total_command)
        input_file = os.path.join(networkpath,input_name)

        # Remove input file
        if os.path.exists(input_file):
            os.remove(input_file)

        if delay is not None:
            sleep(delay)
        startover = False #ensure we use the surface fit from PhaseCam next iteration
        bmc1k_mon.watch(0.1) #this is watching for new awaiting_dm in networkpath


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

def coeffs_to_command(coeffsarray,zernikepolys):
    '''
    From: https://pypi.org/project/zernike/

    Use zernike_scratch.py for testing of demo functions on project page above.

    4D convention / zernike.py
    #1 Piston / Piston
    #2 Tilt x / Tilt x
    #3 Tilt y / Tilt y
    #4 Power / Power
    #5 Astig x / Oblique Astig
    #6 Astig y / Vertical Astig
    #7 Coma x / Vertical Coma
    #8 Coma y / Horizontal Coma
    #9 Primary spherical / Vertical trefoil
    #10 Trefoil x / Oblique Trefoil
    #11 Trefoil y / Primary spherical
    #12 Secondary Astig x / Secondary Vertical Astig 
    #13 Secondary Astig y / Secondary Oblique Astig
    #14 Seconday Coma x / Vertical Quadrafoil
    #15 Secondary Coma y / Oblique Quadrafoil
    #16 Secondary Spherical / Secondary Horizontal Coma
    #17 Tetrafoil x / Secondary Vertical Coma
    #18 Tetrafoil y / Secondary Oblique Trefoil
    #19 Secondary trefoil x / Secondary Vertical Trefoil
    #20 Secondary trefoil y / Oblique Tetrafoil
    #21 Tertiary Astig x / Vertical Trefoil
    #22 Tertiary Astig y / Tertiary Spherical

    TODO need to verify output coeffs ordering from 4D
    TODO need to deduce the units of the output coefficients
    '''
    surface_reconstruction = [poly * coeff for poly, coeff in zip(zernikepolys,coeffsarray)]

    return surface_reconstruction

def zernike_radial(rho, n, m):
    radial_term = np.zeros_like(rho, dtype=float)
    for s in range((n - abs(m)) // 2 + 1):
        coef = (-1) ** s * np.math.factorial(n - s)
        coef /= (
            np.math.factorial(s)
            * np.math.factorial(int((n + abs(m)) / 2) - s)
            * np.math.factorial(int((n - abs(m)) / 2) - s)
        )
        radial_term += coef * rho ** (n - 2 * s)
    return radial_term

def zernike_normalization(n, m):
    # Calculate the normalization factor for Zernike polynomials
    norm_factor = np.sqrt((2 * (n + 1)) / (1 + (m == 0)))
    return norm_factor

def zernike_polynomial(size, n, m):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2 + Y**2)
    
    # Create a circular mask with a slightly larger radius to include 34 elements
    mask = rho <= 1.06  # Adjust the radius as needed
    
    # Calculate the radial component of the Zernike polynomial
    radial_component = zernike_radial(rho, n, m)
    
    if m == 0:
        azimuthal_component = np.ones_like(rho)
    elif m > 0:
        azimuthal_component = np.cos(m * np.arctan2(Y, X))
    else:
        azimuthal_component = np.sin(-m * np.arctan2(Y, X))
    
    # Combine the radial and azimuthal components to get the Zernike polynomial
    zernike = radial_component * azimuthal_component
    
    # Apply the circular mask
    zernike[~mask] = 0.0

    # Normalize the Zernike polynomial
    norm_factor = zernike_normalization(n, m)
    zernike /= norm_factor
    
    return zernike

def generate_zernike_images(size, nm_indices):
    images = []
    for pair in nm_indices:
        n = pair[0]
        m = pair[1]
        zernike_poly = zernike_polynomial(size, n, m)
        images.append(zernike_poly)
    return images



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


class BMC1KMonitor(FileMonitor):
    '''
    Set the DM machine to watch a particular FITS files for
    a modification, indicating a request for a new DM actuation
    state.

    Will ignore the current file if it already exists
    when the monitor starts (until it's modified).
    '''
    def __init__(self, netpath, input_file='awaiting_dm'):
        '''
        Parameters:
            locpath : str
                Local path to create and watch for status files.
            rempath: str
                Remote path to scp status files to.
        '''
        super().__init__(os.path.join(netpath, input_file))

    def on_new_data(self, newdata):
        '''
        On detecting an updated dm_input.fits file,
        load the image onto the DM and scp an
        empty 'dm_ready' file to the 4D machine
        '''
        # Load image from FITS file onto DM channel 0
        log.info('Setting DM from new image file {}'.format(newdata))
        update_status_fname = os.path.join(os.path.dirname(self.file), 'dm_ready')
        open(update_status_fname, 'w').close()
        # to_user = 'PhaseCam'
        # to_address = '192.168.1.3'
        # load_channel(newdata, 1) #dmdisp01

        # Write out empty file locally, then scp over to tell 4Sight the DM is ready.
        # update_status_file(localfpath=local_status_fname,
        #                    remotefpath=self.remote_send,
        #                    user=to_user,address=to_address)



if __name__ == '__main__':
    #Engineering parameters
    kilo_dm_width = 34 #actuators
    n_actuators = 952
    # m_volume_factor = 0.5275
    # m_act_gain = -1.1572
    # m_dm_input = np.sqrt(cmd * m_volume_factor/m_act_gain)
    optimal_voltage_bias = -1.075 #this is the physical displacement in microns for 70% V bias
    #### ---- #### ---- #### ---- #### ----
    save_measure_dir = "C:\\Users\\PhaseCam\\Documents\\jay_4d\\4d-automation\\test"
    reference_flat = "C:\\Users\\PhaseCam\\Documents\\jay_4d\\reference_lamb20avg12_average_ttp-removed.h5"
    if machine_name.upper() == 'PINKY':
        home_folder = "/home/jkueny"
        remote_folder = 'C:\\Users\\PhaseCam\\Desktop\\4d-automation2'
        kilo_map = np.load('/opt/MagAOX/calib/dm/bmc_1k/bmc_2k_actuator_mapping.npy')
        kilo_mask = (kilo_map > 0)
        bias_matrix = optimal_voltage_bias * np.eye(kilo_dm_width**2)[kilo_mask.flatten()]
        # cmds_matrix = optimal_voltage_bias * np.eye(kilo_dm_size[0]*kilo_dm_size[1])[kilo_mask.flatten()]
        # dm_cmds = cmds_matrix.reshape(n_actuators,kilo_dm_size[0],kilo_dm_size[1])
        # single_pokes = []
        # for i in range(len(dm_cmds[0])): #34 length
        #     single_pokes.append(dm_cmds[i])
        #     break #starting with one command for now
        # print(len(single_pokes))
    elif machine_name.upper() == 'PHASECAM':
        home_folder = 'C:\\Users\\PhaseCam'
        remote_folder = "/home/jkueny"
    else:
        print('Error, what machine? Bc apparently it is not pinky or the 4D machine...')
    # kilo_map = np.load('/opt/MagAOX/calib/dm/bmc_1k/bmc_2k_actuator_mapping.npy')
    dm_run(dmglobalbias=bias_matrix,
                    networkpath=remote_folder,
                    coeffsfname='surface_zernikes.npy',
                    reference=reference_flat,
                    dry_run=True,
                    pupildiam=kilo_dm_width,)
