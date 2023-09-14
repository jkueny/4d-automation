import os, sys

import numpy as np
# import h5py

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
#add 4D scripting directory to sys path
#hard code it in for now

try:
    # Hard-coded path to Python scripting library on PhaseCam machine
    sys.path.append('C:\\Program Files (x86)\\4Sight2.24\\scripts')
    ''' 4Sight Python Script '''
    from Scripting.App import *
    from Scripting.Config import *
    from Scripting.Data import *
    from Scripting.Hardware import *
    from Scripting.Measurements import *
    from Scripting.Modify import *
    from Scripting.Show import *
    # connect to 4Sight session (4Sight must be open!)
    # try:
    #     connectionmanager.connect()
    # except core.ZygoError:
    #     log.warning('Zygo library loaded but connection to Mx could not be established.')
except ImportError:
    log.warning( 'Could not load 4Sight Python library!' )
    log.warning( 'Functionality will be severely crippled.')

save_measure_dir = "C:\\Users\\PhaseCam\\Documents\\jay_4d\\4d-automation"

def capture_frame(rawfolder=save_measure_dir,filenameprefix=None, mtype='acquire'):
    '''
    Capture an image on the PhaseCam via 4Sight. Saves to disk.

    Parameters:
        rawfolder : str (TODO optional?)
            Location where frame data is to be stored.
        filenameprefix : str (optional)
            Filename of output. If not provided, 4Sight will
            capture the image and load it into the interface
            but it's up to the user to use the GUI to save
            it out.
        mtype : str
            'acquire' or 'measure'. 'Acquire' takes a measurement
            without analyzing or updating the GUI (faster), while
            'measure' takes a measurement, analyzes, and updates
            the GUI (slower).

    The output is a .h5 file that includes the raw surface 
    (no Zernike modes removed, even if selected in 4Sight), intensity,
    and 4Sight attributes.

    It's expected that all capture parameters will be set manually
    in 4Sight: exposure time, masks, etc.
    '''
    log.info('4Sight: capturing frame and acquiring from camera.')
    num_meas = 1
    if mtype.upper() == 'ACQUIRE':
        num_acq_frames = acquire_frames(num_frames=num_meas, 
                                    output_dir=rawfolder, 
                                    interval=None, 
                                    use_detector_area=True, 
                                    filename_prefix=filenameprefix, 
                                    show_progress=True,
                                    )
        MessageBox("%d of %d frames acquired" % (num_acq_frames, num_meas))

    elif (mtype.upper() == 'MEASURE') or (mtype.upper() == 'MEASUREMENT'):
        measurement = AverageMeasure(frameCount=10,mblk='raw')
    else:
        raise ValueError('Not understood! Did you mean "acquire" or "measure"?')

    if filenameprefix is not None:
        log.info('4Sight: writing out to {0}'.format(filename))
        if not SaveMeasurement(data=measurement,filename=filenameprefix):
            log.warning('Error saving the measurement')

def save_surface(filename):
    '''
    Save the surface currently loaded in 4Sight out
    as a hdf5 file. This may or may not remove Zernikes.
    I need to figure that out.

    Parameters:
        filename : str
            Filename to save surface out to (as a .datx)

    This mostly serves as an example of how to grab data
    from Mx control elements. control_path can be found
    by right-clicking on a GUI element in Mx in choosing
    the "control path"(?) option in the dropdown.

    '''
    control_path = ("MEASURE", "Measurement", "Surface", "Surface Data")
    surface_control = ui.get_control(control_path)
    surface_control.save_data(filename) # .datx?

def read_hdf5(filename, mode='r'):
    '''
    Simple wrapper around h5py to load a file
    up (just because I have a hard time remembering
    the syntax).

    Parameters:
        filename : str
            File to open
        mode : str (optional)
            Mode to open file in.
    Return : nothing
    '''
    return h5py.File(filename, mode)

def open_in_Mx(filename):
    '''
    Open a .datx file in Mx

    Parameters:
        filename : str
            File path to open.
    Returns: nothing
    '''
    mx.load_data(filename)