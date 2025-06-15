
import argparse
import pathlib
import json

import torch

import data
import runner
import webui
import configuration

def train_with_memory_debug():
    torch.cuda.memory._record_memory_history()
    try:
        train_()
    finally:
        torch.cuda.memory._dump_snapshot("out_memory.pickle")

def train(config):
    config.training = True
    
    run = runner.Runner(config)
    run.init_and_load()

    run.train()

def evaluate(config):
    config.training = False
    config.checkpoint_load = True
    config.testdata_size = config.testdata_size_max
    
    run = runner.Runner(config)
    run.init_and_load()

    loss = run.compute_test_loss()
    loss_nopacks = run.compute_test_loss(skip_packs=True)

    w = max(map(len, loss))
    print('\nLoss:')
    for k, val in loss.items():
        print(f'  {k:{w}}  {val:7.4f}  {loss_nopacks[k]:7.4f}')

def output_sample_draft(args):
    tokens, tokens_list = load_tokens()
    
    config = Config(len(tokens))
    init_common(config)
    
    dataset_train, dataset_test = load_dataset(config)
    
    model = Decoder(config)
    load_checkpoint(model, args, config)
    model.to(config.device)
    model.eval()

    generator = torch.Generator().manual_seed(args.seed)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.count, shuffle=False, pin_memory=True, generator=generator)

    count = min(args.count, len(loader_test))

    picks, packs = next(iter(loader_test))
    picks = picks.to(config.device)
    packs = packs.to(config.device)

    logits = model(picks, packs=packs)
    logits.to('cpu')

    w = max(map(len, tokens_list))

    for it in range(logits.shape[0]):
        logits_it = logits[it]
        for i in range(1,config.seq_len):
            count = torch.sum(logits_it[i-1] != float('-inf')).item()
            if count > config.seq_pack: count = 8
            values, indices = torch.topk(logits_it[i-1], k=count, sorted=True, largest=True)
            
            print(f'{tokens_list[picks[it,i]]:{w}} {logits_it[i-1,picks[it,i]]:4.1f} | ' + ', '.join(
                f'{tokens_list[tok]} {p:.1f}' for tok, p in zip(indices, values)
            ))
        print()

def save_testdata(config):
    config.testdata_size = config.testdata_size_max
    config.batch_size = config.testdata_size

    with open('datasets.json') as f:
        datasets_json = json.load(f)
    
    run = runner.Runner(config)
    run.init()
    run.load_dataset(datasets_json=datasets_json)
    run.load_loader()

    tensors = next(iter(run.loader_test))
    torch.save(tensors, config.testdata_only_path)
    print(f'Wrote test dataset to {config.testdata_only_path}')

    with open('datasets_ext.json', 'w') as f:
        json.dump(datasets_json, f)
    print(f'Wrote datasets_ext.json')
        
def main(cmd_args=None):
    parser = argparse.ArgumentParser()

    for cls in configuration.Config.mro():
        if cls is object: continue
        group_name = cls.__name__[7:]
        group = parser.add_argument_group(group_name)
        for name, typ in cls.__annotations__.items():
            mtyp = typ
            if mtyp.__name__ == 'Optional': mtyp = mtyp.__args__[0]
            if name[0] == '_': continue
            action = None
            if mtyp == bool:
                action = argparse.BooleanOptionalAction
            group.add_argument('--' + name.replace('_', '-'), type=mtyp, dest='config_'+name, metavar=mtyp.__name__, action=action)
    
    parser.add_argument('--config', type=pathlib.Path, action='append')
    
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('init_tokens').add_argument('path', type=pathlib.Path)
    subparsers.add_parser('convert').add_argument('paths', type=pathlib.Path, nargs='+')
    
    parser_train = subparsers.add_parser('train')

    parser_sample = subparsers.add_parser('sample')
    parser_sample.add_argument('--seed', type=int, default=0)
    parser_sample.add_argument('--count', type=int, default=4)

    parser_hyperopt = subparsers.add_parser('hyperopt')

    parser_getjson = subparsers.add_parser('getjson')
    parser_getjson = subparsers.add_parser('gen_token_meta')
    
    subparsers.add_parser('webui')
    subparsers.add_parser('evaluate')
    subparsers.add_parser('save_testdata')

    parser_logtocsv = subparsers.add_parser('log_to_csv')
    parser_logtocsv.add_argument('path', nargs='?', default=None)
    
    args = parser.parse_args(cmd_args)
    
    config = configuration.Config()
    args_config = {}
    for i in config.keys():
        val = getattr(args, 'config_' + i, None)
        if val is not None:
            args_config[i] = val

    if args.config is not None:
        for path in args.config:
            with path.open() as f:
                config.merge(json.load(f))

    config.merge(args_config)
    config._args_config = args_config

    if args.command == 'init_tokens':
        data.init_tokens(args.path)
    elif args.command == 'convert':
        process_files(args.paths)
    elif args.command == 'train':
        train(config)
    elif args.command == 'sample':
        output_sample_draft(args)
    elif args.command == 'hyperopt':
        import hyperopt
        hyperopt.hyperopt(config)
    elif args.command == 'log_to_csv':
        import hyperopt
        hyperopt.log_to_csv(args.path)
    elif args.command == 'getjson':
        data.download_json()
    elif args.command == 'gen_token_meta':
        data.generate_token_meta()
    elif args.command == 'webui':
        webui.main(config)
    elif args.command == 'evaluate':
        evaluate(config)
    elif args.command == 'save_testdata':
        save_testdata(config)

if __name__ == '__main__':
    main()
