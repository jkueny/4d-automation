import glob as glob
import os
from time import sleep
import numpy as np
import subprocess
import platform
# from zernike import RZern
# from scipy.linalg import hadamard
import h5py

current_platform = platform.system()

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

def dm_run( dm_inputs,
            dmglobalbias,
            networkpath,
            remotepath,
            outname,
            delay=None,
            consolidate=True,
            dry_run=True,
            clobber=False,
            pupildiam=34,
            # unitcirclex=621,
            # unitcircley=642,
            # unitcirclerad=331,
            status_name='dm_ready',
            input_name='dm_input.fits',
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
        networkpath : str
            Path to folder where cross-machine communication 
            will take place. Both machines must have read/write 
            privileges.
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

    # if not (dry_run or clobber):
    #     # Create a new directory outname to save results to
    #     assert not os.path.exists(outname), '{0} already exists!'.format(outname)
    #     os.mkdir(outname)

    #Generate the Zernike polynomials now, to avoid doing it in a loop.
    #Because the 4D uses a weird ordering scheme, hard-coding this indexing
    #for now. Starting with the first 16 polynomials in Zemax(?) ordering.
    # nm_pairs = [(0,0),(1,1),(1,-1),(2,0),(2,2),(2,-2),(3,1),(3,-1),
    #         (4,0),(3,3),(3,-3),(4,2),(4,-2),(5,1),(5,-1),(6,0)]
    # zernike_polys = generate_zernike_images(kilo_dm_width, nm_pairs)
    # Calculate the coordinates of the upper-left and lower-right corners of the square bounding box
    # upper_left_x = unitcirclex - unitcirclerad
    # upper_left_y = unitcircley - unitcirclerad
    # lower_right_x = unitcirclex + unitcirclerad
    # lower_right_y = unitcircley + unitcirclerad
    # circle_crop_sz = unitcirclerad*2 + 1
    # surface_basis = generate_zernike_basis(norder=15,imgsize=circle_crop_sz,npolys=120)
    # command_basis = generate_zernike_basis(norder=15,imgsize=pupildiam,npolys=120)
    print('Initiating FileMonitor...')
    bmc1k_mon = BMC1KMonitor(networkpath)
    #start from fresh, this zero array added to the optimal bias command.
    for idx, inputs in enumerate(dm_inputs):
        old_files = glob.glob(os.path.join(networkpath,'dm_input*.fits'))
        for old_file in old_files:
            if os.path.exists(old_file):
                os.remove(old_file)
        # Write out FITS file with requested DM input
        log.info('Setting DM to state {0}/{1}.'.format(idx + 1, len(dm_inputs)))
        inputs += dmglobalbias
        if not dry_run:
            dm01.write(inputs)
        # input_file = os.path.join(networkpath,input_name)
        # write_fits(filename=input_file, data=inputs, dtype=np.float32, overwrite=True)
        # log.info('Sending new command...')
        # total_command = dmglobalbias - surface_fit
        #I don't think we will save the commands as files for this job, commenting below.
        # #Remove any old inputs if they exist
        # old_files = glob.glob(os.path.join(networkpath,'dm_input*.fits'))
        # for old_file in old_files:
        #     if os.path.exists(old_file):
        #         os.remove(old_file)
        # Write out FITS file with requested DM input
        # input_file = os.path.join(networkpath,status_name)

        # # Remove input file
        # if os.path.exists(input_file):
        #     os.remove(input_file)

        if delay is not None:
            sleep(delay)
        if idx == 0: #the Monitor class will take care of this from here
            open('dm_ready','w').close()
            update_status_fname = 'dm_ready'
            # open(update_status_fname, 'w').close()
            to_user = 'PhaseCam'
            to_address = '192.168.1.3'
            # load_channel(newdata, 1) #dmdisp01

            # Write out empty file locally, then scp over to tell 4Sight the DM is ready.
            update_status_file(localfpath=update_status_fname,
                            remotefpath=remotepath,
                            user=to_user,address=to_address)
        # open(os.path.join(networkpath,'dm_ready'),'w').close()

        bmc1k_mon.watch(0.1) #this is watching for new awaiting_dm in networkpath
    if consolidate:
        log.info('Writing to consolidated .hdf5 file.')
        # Consolidate individual frames and inputs
        # Don't read attributes into a dictionary. This causes python to crash (on Windows)
        # when re-assignging them to hdf5 attributes.
        alldata = read_many_raw_h5(sorted(glob.glob(os.path.join(outname,'frame_*.h5'))), 
                                     attrs_to_dict=True, mask_and_scale=True)
        write_dm_run_to_hdf5(os.path.join(outname,'alldata.hdf5'),
                             np.asarray(alldata['surface']),
                             alldata['surface_attrs'][0],
                             np.asarray(alldata['intensity']),
                             alldata['intensity_attrs'][0],
                             alldata['attrs'][0],
                             np.asarray(dm_inputs),
                             alldata['mask'][0]
                             )

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
        # print('Attempting to send to {}'.format(send_to))
        print('Attempting to send {}'.format(localfpath))
        subprocess.run(['scp', localfpath, send_to], check=True)
        print('File copied to {}'.format(send_to))
    except subprocess.CalledProcessError as e:
        print('Error: {}'.format(e))

def grab_data_hdf5(infilename,groupname,datasetname,dataname):
    hdf5_file = h5py.File(infilename, 'r')  # 'r' for read-only mode

    group = hdf5_file['measurement0']  # Replace 'group_name' with the actual group name
    dataset = group['genraw']  # Replace 'dataset_name' with the actual dataset name

    #Convert the data to a numpy array and flatten it
    data = dataset['data']
    attributes = dataset.attrs
    data = data[:]
    data[data > 2.0] = 0.
    return data

def compute_command(surfcoeffs,zernbasis,dmsize):
    basis_matrix = zernbasis.copy()
    for i in range(num_polynomials):
        basis_matrix[:,i] = basis_matrix[:,i] * surfcoeffs[i]

    # Sum all the scaled images to create a single image
    summed_image = np.sum(basis_matrix, axis=1)

    # Reshape the summed image to the original image size
    original_image_size = (dmsize,dmsize)
    summed_image = summed_image.reshape(original_image_size)
    return summed_image
    

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

def generate_zernike_basis(norder, imgsize, npolys=120):
    cart = RZern(15)
    L, K = imgsize, imgsize
    ddx = np.linspace(-1.0, 1.0, K)
    ddy = np.linspace(-1.0, 1.0, L)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    zernike_matrix = np.empty((L*K, npolys))
    c = np.zeros(cart.nk)
    for i in range(npolys):
        c *= 0.0
        c[i] = 1.0
        Phi = cart.eval_grid(c, matrix=True)
        Phi[Phi != Phi] = 0.
        flattened_zernike = Phi.flatten()
        zernike_matrix[:, i] = flattened_zernike
    return zernike_matrix



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
        log.info('Watching for changes made to this file {0}'.format(os.path.join(netpath, input_file)))
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
        # open(update_status_fname, 'w').close()
        to_user = 'PhaseCam'
        to_address = '192.168.1.3'
        # load_channel(newdata, 1) #dmdisp01

        # Write out empty file locally, then scp over to tell 4Sight the DM is ready.
        update_status_file(localfpath=update_status_fname,
                           remotefpath=self.remote_send,
                           user=to_user,address=to_address)

def parse_raw_h5(filename, attrs_to_dict=True, mask_and_scale=False):
    '''
    Given a .datx file containing raw surface measurements,
    return a dictionary of the surface and intensity data,
    as well as file and data attributes.
    
    Parameters:
        filename : str
            File to open and raw (.datx)
        attrs_to_dict : bool, opt
            Cast h5py attributes objects to dicts
        mask_and_scale : bool, opt
            Mask out portion of surface/intensity
            maps with no data and scale from wavefront
            wavelengths to surface microns.

    Returns: dict of surface, intensity, masks, and attributes

    I really dislike this function, but the .datx files are a mess
    to handle in h5py without a wrapper like this. 
    '''
    
    h5file = h5py.File(filename, 'r')
    
    assert 'measurement0' in list(h5file.keys()), 'No "Measurement" key found. Is this a raw .h5 file?' 
    
    # Get surface and attributes
    data = h5file['measurement0']['genraw']['data']
    data = data[:]
    data[data > 10.] = 0.
    surface = data
    surface_attrs = h5file['measurement0']['genraw']['data'].attrs
    # Define the mask from the "no data" key
    # mask = np.ones_like(surface).astype(bool)
    # mask[surface == surface_attrs['No Data']] = 0
    # Mask the surface and scale to surface in microns if requested
    # if mask_and_scale:
    #     surface[~mask] = 0
    #     surface *= surface_attrs['Interferometric Scale Factor'][0] * surface_attrs['Wavelength'] * 1e6
    
    # Get file attributes (overlaps with surface attrs, I believe)

    attrs = h5file['Measurement']['Attributes'].attrs
    
    # Get intensity map
    intensity = h5file['Measurement']['Intensity'].value
    intensity_attrs = h5file['Measurement']['Intensity'].attrs

    if attrs_to_dict:
        surface_attrs = dict(surface_attrs)
        attrs = dict(attrs)
        intensity_attrs = dict(intensity_attrs)

    h5file.close()
    
    return {
        'surface' : surface,
        'surface_attrs' : surface_attrs,
        # 'mask' : mask,
        # 'intensity' : intensity,
        # 'intensity_attrs' : intensity_attrs,
        # 'attrs' : attrs 
    }

def read_many_raw_h5(filenames, attrs_to_dict=True, mask_and_scale=False):
    '''
    Simple loop over many .datx files and consolidate into a list
    of surfaces, intensity maps, and attributes 

    Parameters:
        filenames: list
            List of strings pointing to filenames
        attrs_to_dict : bool, opt
            Cast h5py attributes objects to dicts
        mask_and_scale : bool, opt
            Mask out portion of surface/intensity
            maps with no data and scale from wavefront
            wavelengths to surface microns. 

    Returns: list of dicts. See parse_raw_datx.
    '''
    consolidated = {
        'surface' : [],
        'surface_attrs' : [],
        # 'intensity' : [],
        # 'intensity_attrs' : [],
        # 'attrs' : [],
        # 'mask' : [],
    }
    for f in filenames:
        fdict = parse_raw_h5(f, attrs_to_dict=attrs_to_dict, mask_and_scale=mask_and_scale)
        for k in fdict.keys():
            consolidated[k].append(fdict[k])
    return consolidated

# def get_hadamard_modes(Nact):
#     np2 = 2**int(np.ceil(np.log2(Nact)))
#     #print(f'Generating a {np2}x{np2} Hadamard matrix.')
#     hmat = hadamard(np2)
#     return hmat#[:Nact,:Nact]


if __name__ == '__main__':
    #Engineering parameters
    kilo_dm_width = 34 #actuators
    n_actuators = 952
    # m_volume_factor = 0.5275
    # m_act_gain = -1.1572
    # m_dm_input = np.sqrt(cmd * m_volume_factor/m_act_gain)
    optimal_voltage_bias = -1.075 #this is the physical displacement in microns for 70% V bias
    #### ---- #### ---- #### ---- #### ----
    home_folder = "/home/jkueny"
    remote_folder = 'C:\\Users\\PhaseCam\\Desktop\\4d-automation'
    shared_folder = '/home/jkueny/netshare/4d-automation2'
    kilo_map = np.load('/opt/MagAOX/calib/dm/bmc_1k/bmc_2k_actuator_mapping.npy')
    kilo_mask = (kilo_map > 0)
    # bias_matrix = optimal_voltage_bias * np.eye(kilo_dm_width**2)[kilo_mask.flatten()]
    bias_matrix = optimal_voltage_bias + np.zeros((kilo_dm_width,kilo_dm_width))
    cmds_matrix = 0.015 * np.eye(kilo_dm_width*kilo_dm_width)[kilo_mask.flatten()]# 15 nm
    dm_cmds = cmds_matrix.reshape(n_actuators,kilo_dm_width,kilo_dm_width)
    # dm_cmds = bias_matrix
    single_pokes = []
    for i in range(n_actuators):
        single_pokes.append(dm_cmds[i])
    #     break #starting with one command for now
    print(f'TODO: {len(single_pokes)} DM pokes.')
    # kilo_map = np.load('/opt/MagAOX/calib/dm/bmc_1k/bmc_2k_actuator_mapping.npy')
    dm_run( dm_inputs=dm_cmds,
            dmglobalbias=bias_matrix,
            networkpath=shared_folder,
            remotepath=remote_folder,
            outname=f'{shared_folder}/data',
            dry_run=False,
            pupildiam=kilo_dm_width,
            )
