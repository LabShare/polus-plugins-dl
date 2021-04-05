
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
            
        self.sp['base_lr'] = '1.0E-4'
        self.sp['display'] = '1'
        self.sp['max_iter'] = '100'
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


    def write(self, filepath, test_iter, iterations):
        """
        Export solver parameters to INPUT "filepath".
        
        Args:
        filepath: filepath to save solver.prototxt.
        test_iter: total number of validation files.
        iterations: number of training iterations.
        """

        f = open(filepath, 'w')
        self.sp['test_iter'] = test_iter
        self.sp['max_iter'] = iterations
        self.sp['snapshot'] = iterations
        for key, value in (self.sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
