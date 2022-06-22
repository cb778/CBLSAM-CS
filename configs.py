def config():
    configSetting = {
        'dataset_name': 'Dataset',
        'train_name': 'train.name.h5',
        'train_api': 'train.apiseq.h5',
        'train_tokens': 'train.tokens.h5',
        'train_desc': 'train.desc.h5',
        'valid_name': 'test.name.h5',
        'valid_api': 'test.apiseq.h5',
        'valid_tokens': 'test.tokens.h5',
        'valid_desc': 'test.desc.h5',
        'name_len': 6,
        'api_len': 30,
        'tokens_len': 50,
        'desc_len': 30,
        'vocab_size': 10000,
        'vocab_name': 'vocab.name.json',
        'vocab_api': 'vocab.apiseq.json',
        'vocab_tokens': 'vocab.tokens.json',
        'vocab_desc': 'vocab.desc.json',
        'batch_size': 128,
        'Epoch': 15,
        'learning_rate': 1e-4,
        'adam_epsilon': 1e-8,
        'warmup_steps': 5000,
        'fp16': False,
        'd_word_dim': 128, #
        'd_model':128 * 2 ,
        'd_ffn': 512,   # use in  FeedForward
        'n_heads': 8,
        'n_layers': 1,
        'd_k': 16 * 2 ,
        'd_v': 16 * 2,
        'pad_idx': 0,
        'margin': 0.3986,
        'sim_measure': 'cos',
        'emb_size':128, # the dimension is embedding dimension,and must equal d_word_dim
        'hidden_size': 128,  # bidirectional=true, so *2 = 256


    }
    return configSetting
