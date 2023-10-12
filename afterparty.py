import h5py
import os, glob

basedir = os.environ['HOME']
workdir = f'{basedir}/netshare/4d-automation2'
datadir = f'{workdir}/singles_20231012'

def parse_raw_h5(filename, attrs_to_dict=False, mask_and_scale=False):
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

    # attrs = h5file['Measurement']['Attributes'].attrs
    
    # Get intensity map
    # intensity = h5file['Measurement']['Intensity'].value
    # intensity_attrs = h5file['Measurement']['Intensity'].attrs

    if attrs_to_dict:
        surface_attrs = dict(surface_attrs)
        attrs = dict(attrs)
        intensity_attrs = dict(intensity_attrs)

    h5file.close()
    
    return surface

    # return {
    #     'surface' : surface,
    #     # 'surface_attrs' : surface_attrs,
    #     # 'mask' : mask,
    #     # 'intensity' : intensity,
    #     # 'intensity_attrs' : intensity_attrs,
    #     # 'attrs' : attrs 
    # }

for each,h5file in enumerate(sorted(glob.glob(f'{datadir}frame*.h5'))):
    surf_arr = parse_raw_h5(h5file)
    