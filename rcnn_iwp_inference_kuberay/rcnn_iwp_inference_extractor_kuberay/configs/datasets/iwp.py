# custom configs for iwp dataset

model = dict(
    proposal_generator=dict(
        anchor_generator=dict(sizes=[[32], [64], [128], [256], [512]]),
    ),
    roi_heads=dict(
        num_classes=1,
    ),
    backbone=dict(
        net=dict(img_size=1024, in_chans=3),
        square_pad=1024,
    ),
    # Use RGB format, will be converted to BGR later if needed
    pixel_mean=[122.46, 90.90, 108.14],  # RGB
    pixel_std=[33.94, 31.38, 28.77],  # RGB
)

dataloader = dict(
    train=dict(
        dataset=dict(
            names="iwp_train",
        ),
    ),
    test=dict(
        dataset=dict(
            names="iwp_val",
        ),
    ),
    evaluator=dict(
        max_dets_per_image=100,
    ),
)
