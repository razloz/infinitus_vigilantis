#!./.env/bin/python3
"""Memory tuning for IVy Cauldron transformer."""
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2023, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'muenster'


if __name__ == '__main__':
    import torch
    from source.ivy_commons import ivy_dispatcher
    from source.ivy_cauldron import Cauldron
    DEVICE_TYPE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if DEVICE_TYPE == 'cpu':
        print('CUDA not found. Exiting...')
        exit()
    DEVICE = torch.device(DEVICE_TYPE)
    allocated = torch.cuda.memory_allocated
    possible_threads = 0
    cauldrons = list()
    num_cauldrons = 0
    mem_usage = 0
    while True:
        try:
            cauldrons.append(Cauldron(verbosity=1))
        except:
            mem_usage = allocated(device=DEVICE)
            num_cauldrons = len(cauldrons)
            print('mem_usage:', mem_usage)
            print('num_cauldrons:', num_cauldrons)
            break
    max_cauldrons = int(num_cauldrons * 0.2)
    print('Set max_cauldrons to', max_cauldrons)
    cauldrons = cauldrons[:max_cauldrons]
    processes = list()
    max_cauldrons = 0
    for cauldron in cauldrons:
        try:
            processes.append(
                ivy_dispatcher(
                    cauldron.train_network,
                    ftype='process',
                    kwargs=dict(hours=1e-5, checkpoint=1),
                    ),
                )
            max_cauldrons += 1
        except:
            pass
    for process in processes:
        process.join()
    print('Adjusted max_cauldrons to', max_cauldrons)
    exit()
    threads = list()
    max_cauldrons = 0
    for cauldron in cauldrons:
        try:
            threads.append(
                ivy_dispatcher(
                    cauldron.train_network,
                    ftype='thread',
                    kwargs=dict(hours=1e-5, checkpoint=1),
                    ),
                )
            max_cauldrons += 1
        except:
            pass
    for thread in threads:
        thread.join()
    print('Adjusted max_cauldrons to', max_cauldrons)
