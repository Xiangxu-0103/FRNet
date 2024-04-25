lr = 0.01
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6))

param_scheduler = [
    dict(
        type='OneCycleLR',
        total_steps=50000,
        by_epoch=False,
        eta_max=lr,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=100.0)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=50000, val_interval=1000)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(enable=False, base_batch_size=16)
