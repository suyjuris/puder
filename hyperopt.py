
import csv
import json
import datetime

import torch
import nevergrad as ng

import runner

def _train(run, x):
    xx = {k:v for k,v in x.items() if k[0] != '_'}
    run.config.merge(xx)
    run.load_model()
    run.train()
    return run.compute_test_loss()

def _worker(run, device_it, queue_params, queue_results):
    run.config.device = f'cuda:{device_it}'
    run.config.silent = True

    with torch.cuda.device(device_it):
        run.load_loader()

        while True:
            param = queue_params.get()
            if param is None: break
            loss = _train(run, param)
            queue_results.put((param, loss))

def get_current_time():
    return datetime.datetime.now().isoformat().replace('T', '_').replace(':', '-')[:19]
            
def hyperopt(config):
    config.merge({
        'testdata_size': 1024,
        'train_duration_max': 400,
        'train_instances_max': 100100,
        'checkpoint_load': False,
        #'checkpoint_load': True,
        #'checkpoint_load_path': pathlib.Path('models/model_2025-05-01_21-36-47.pt'),
        'checkpoint_save': False,
        'tensorboard_embedding': False,
        'period_test': -1,
        'train_cutoffs': {(100, 1.5), (240, 1.1)},
    })
    config.training = True
    run = runner.Runner(config)
    run.init()
    run.load_dataset()

    param = ng.p.Dict(
        hidden_layers = ng.p.TransitionChoice(range(3,8+1)),
        head_count = ng.p.TransitionChoice(range(2,14+1,2)),
        head_size = ng.p.TransitionChoice(range(8,48+1,8)),
        pack_hidden_layers = ng.p.TransitionChoice(range(1,4+1)),
        pack_head_count = ng.p.TransitionChoice(range(2,14+1,2)),
        pack_head_size = ng.p.TransitionChoice(range(8,48+1,8)),
        pack_insert_layer = ng.p.TransitionChoice(range(4)),
        learning_rate = ng.p.Log(lower=1e-5, upper=1e-2, init=0.0038),
        adam_beta1_inv = ng.p.Log(lower=0.01, upper=0.4, init=0.09),
        adam_beta2_inv = ng.p.Log(lower=1e-5, upper=0.01, init=0.00033),
        grad_clip = ng.p.Log(lower=0.1, upper=10.0, init=0.75),
        lr_warmup = ng.p.TransitionChoice([8,16,32,64,128]),
    )
    #param = ng.p.Dict(
    #    learning_rate = ng.p.Log(lower=1e-7, upper=1e-3, init=1e-4),
    #    adam_beta1_inv = ng.p.Log(lower=0.01, upper=0.2, init=0.16),
    #    adam_beta2_inv = ng.p.Log(lower=1e-5, upper=0.01, init=0.001),
    #    grad_clip = ng.p.Log(lower=0.1, upper=10.0, init=0.5),
    #    lr_warmup = ng.p.TransitionChoice([0,8,32,128]),
    #)
    
    n_devices = torch.accelerator.device_count()
    assert n_devices > 0

    budget = 362
    optimizer = ng.optimizers.NgIohTuned(parametrization=param, budget=budget, num_workers=n_devices)

    def map_param(x):
        xx = {}
        for k,v in x.items():
            if k.endswith('_inv'):
                xx[k[:-4]] = 1-v
            else:
                xx[k] = v
        for pre in ['', 'pack_']:
            xx[pre+'hidden_size'] = xx[pre+'head_count'] * xx[pre+'head_size']
            xx[pre+'dense_size'] = 2 * xx[pre+'hidden_size']
        param['_time'] = get_current_time()
        return xx
    
    def constraint(x):
        nparams = 0
        x = map_param(x)
        for pre in ['', 'pack_']:
            one_attention = 3 * x[pre+'hidden_size']**2
            one_dense = 2 * x[pre+'dense_size'] * x[pre+'hidden_size']
            nparams += x[pre+'hidden_layers'] * (one_attention + one_dense)
        nparams_max = 100 * 1000**2
        return float(nparams_max - nparams) + 1000.0 * max(x['pack_insert_layer'] - x['hidden_layers'], 0)
    optimizer.parametrization.register_cheap_constraint(constraint)

    f = open('hyperopt_log.json', 'a')

    hyp_id = get_current_time()

    best = [float('inf'), None, None]
    
    def process(param, result):
        data = {'param': param, 'loss': result, 'hyp_id': hyp_id}
        data_s = json.dumps(data)
        print(data_s, file=f, flush=True)

        if result['total'] < best[0]:
            print(f'New incumbent ({best[0]} -> {result["total"]}):')
            print(json.dumps(param, indent='  '))
            print(json.dumps(result, indent='  '))
            best[0] = result['total']
            best[1] = param
            best[2] = result
            
        return result['total']

    if n_devices == 1:
        run.load_loader()
        def train(x):
            xx = map_param(x)
            return process(xx, _train(run, xx))
        try:
            final = optimizer.minimize(train)
        except KeyboardInterrupt:
            print('Interrupted')
        
    else:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn')

        queue_params = mp.SimpleQueue()
        queue_results = mp.SimpleQueue()

        stored = {'id': 0}
        
        def push():
            cand = optimizer.ask()
            stored[stored['id']] = cand
            param = map_param(*cand.args)
            param['_cand_id'] = stored['id']
            queue_params.put(param)
            stored['id'] += 1
        def pop():
            param, result = queue_results.get()
            print(f'[{param["_cand_id"]+1:3d}/{budget}] {result["total"]}')
            value = process(param, result)
            cand = stored.pop(param['_cand_id'])
            optimizer.tell(cand, value)

        workers = []
        for it in range(n_devices):
            p = mp.Process(target=_worker, args=(run, it, queue_params, queue_results))
            p.start()
            workers.append(p)
            push()

        for i in range(budget - n_devices):
            pop()
            push()
        for it in range(n_devices): queue_params.put(None)
        for it in range(n_devices): pop()

        final = optimizer.recommend()

        for i in workers:
            i.join()
            
    print('Best:', final)
    print(json.dumps(best[1], indent='  '))
    print(json.dumps(best[2], indent='  '))

def log_to_csv(path=None):
    if path is None:
        path = 'hyperopt_log.json'

    def get_values(data):
        for i, val in data.items():
            if isinstance(val, dict):
                pre = '' if i == 'param' else i+'_'
                for j, v in val.items():
                    yield pre+j, v
            else:
                 yield i, val

    table = [None]
    headers = {}
    f = open(path)
    for line in f:
        data = json.loads(line)
        row = {}
        for i, val in get_values(data):
            it = headers.setdefault(i, len(headers))
            row[it] = str(val)
        table.append(row)
    f.close()
    table[0] = {it:i for i,it in headers.items()}
    
    f_out = open('hyperopt_log.csv', 'w')
    writer = csv.writer(f_out, delimiter=',')
    for row in table:
        r = [''] * len(headers)
        for it, val in row.items():
            r[it] = val
        writer.writerow(r)
    f_out.close()
