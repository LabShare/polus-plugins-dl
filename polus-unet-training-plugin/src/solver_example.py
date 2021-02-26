
class CaffeSolver:

    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default
    values and can export a solver parameter file.
    Note that all parameters are stored as strings. Strings variables are
    stored as strings in strings.
    """

    def __init__(self,
                 trainnet_prototxt_path="model.prototxt", debug=False):

        self.sp = {}
        self.sp['test_iter'] = '12'
        self.sp['test_interval'] = '10'
            
        # critical:
        self.sp['base_lr'] = '1.0E-4'
        # looks:
        self.sp['display'] = '1'
        # pretty much never change these.
        self.sp['max_iter'] = '100'
        # learning rate policy
        self.sp['lr_policy'] = '"fixed"'
        self.sp['momentum'] = '0.9'

        self.sp['snapshot'] = '100'
        self.sp['snapshot_prefix'] = '"snapshot"'  
        self.sp['solver_mode'] = 'CPU'
        self.sp['debug_info'] = 'false'

        self.sp['net'] = '"' + trainnet_prototxt_path + '"'
        self.sp['snapshot_format'] = 'HDF5'
        self.sp['momentum2'] = '0.999'
        self.sp['type'] = '"Adam"'


    def write(self, filepath, test_iter):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        f = open(filepath, 'w')
        self.sp['test_iter'] = test_iter
        for key, value in (self.sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
