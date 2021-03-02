import sys


def process_arguments(argument_list):
    if not argument_list[0].startswith('-'): raise KeyError('Undefined argument: '+argument_list[0])
    # assert args[0].startswith('-')

    known_keys = ['h','i','o','s','d','n','l','t','v']
    help_msg = """usage: python 3dSimulation.py [opt1] [arg1] [opt2] [arg2] ...
    -h\t: print help message
    -i dir\t: input directory to load input (optional). If not provided, default scene is created and not saved in files. 
    \t  If provided, but empty, default scene is created and saved in the directory. 
    -o dir\t: output directory (optional). If not provided, output files are not saved. If not empty, recreated as empty.
    -d N\t: size of the detector grid. If 1 number is passed, grid is NxN. If 2 numbers are passed, grid is N1xN2.
    -s S\t: voxel size of the scene. If 1 number is passed, 3d array's shape is SxSxS. If 3 numbers are passed, shape is S1xS2xS3
    -n num\t: number of gradient steps. If not provided, default is 50
    -l lr\t: learning rate. If not provided, default is calculated from number of detectors and voxel shape, close to 0.001
    -t thr\t: threshold on the voxel value for the final 3d model. If not provided, default is 0.4
    -v [hist [id]] [vox]\t: whether to visualise the output. If 'hist' key provided, the id detector is visualised 
    \t\t\t  (if id not provided, id=1). If 'vox' key provided, 3d voxel models are visualised using matplotlib.
    """
    arguments = {}
    key = 'u'
    for arg in argument_list:
        if arg.startswith('-'):
            key = arg[1:]
            if key not in known_keys: raise KeyError('Undefined argument: -' + key)
            arguments[key] = []
        else:
            arguments[key].append(arg)
    if 'u' in arguments.keys(): raise KeyError('Undefined argument: '+argument_list[0])
    # print('n_arguments:',len(arguments))
    # for key, arg in arguments.items():
    #     print(key,'\n',arg)

    # print('\n\n')
    input_dir = None
    output_dir = None
    det_nx, det_ny = 3, 3
    scene_shape = (32,32,32)
    num_steps = 50
    lr = 0.06 / det_nx / scene_shape[2]
    thresh = 0.4
    det_id = 1
    vis_hist = False
    vis_vox = False

    if 'h' in arguments.keys():
        print(help_msg)
        exit()
    if 'i' in arguments.keys():
        if len(arguments['i']) < 1: raise ValueError('Missing argument for input dir')
        input_dir = arguments['i'][0]
        if len(arguments['i']) > 1: raise ValueError('Too many arguments for input dir')
    if 'o' in arguments.keys():
        if len(arguments['o']) < 1: raise ValueError('Missing argument for output dir')
        output_dir = arguments['o'][0]
        if len(arguments['o']) > 1: raise ValueError('Too many arguments for output dir')
    if 'd' in arguments.keys():
        if len(arguments['d']) == 1: det_nx, det_ny = int(arguments['d'][0]), int(arguments['d'][0])
        elif len(arguments['d']) == 2: det_nx, det_ny = int(arguments['d'][0]), int(arguments['d'][1])
        else: raise ValueError('Wrong arguments for the detector grid size')
    if 's' in arguments.keys():
        if len(arguments['s']) == 1: scene_shape = (int(arguments['s'][0]),int(arguments['s'][0]),int(arguments['s'][0]))
        elif len(arguments['s']) == 3: scene_shape = (int(arguments['s'][0]),int(arguments['s'][1]),int(arguments['s'][2]))
        else: raise ValueError('Wrong arguments for the voxel model size')
        lr = 0.06 / det_nx / scene_shape[2]
    if 'n' in arguments.keys():
        if len(arguments['n']) == 1: num_steps = int(arguments['n'][0])
        else: raise ValueError('Wrong arguments for the number of gradient steps')
    if 'l' in arguments.keys():
        if len(arguments['l']) == 1: lr = float(arguments['l'][0])
        else: raise ValueError('Wrong arguments for the learning rate')
    if 't' in arguments.keys():
        if len(arguments['t']) == 1: thresh = float(arguments['t'][0])
        else: raise ValueError('Wrong arguments for the threshold')
    if 'v' in arguments.keys():
        if not len(arguments['v']): raise ValueError('Missing argument for visualisation')
        if 'hist' in arguments['v']: vis_hist = True
        if 'vox' in arguments['v']: vis_vox = True
        if 'hist' in arguments['v'] and arguments['v'][-1]!='hist':
            det_id = ([int(arguments['v'][i+1]) for i,arg in enumerate(arguments['v']) if arg=='hist'])[0]
    return input_dir, output_dir, det_nx, det_ny, scene_shape, num_steps, lr, thresh, det_id, vis_hist, vis_vox


params = None
if len(sys.argv) > 1: params = process_arguments(sys.argv[1:])
print(params)
