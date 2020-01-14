# from transformers import AlbertModel, AlbertTokenizer


from experiments.predefined import update
from experiments.predefined.base_wiki import env, base_config, siamese_config, joint_config, bert_base_config, \
    roberta_base_config, xlnet_base_config, seq512_config, seq256_config, seq128_config, seq64_config, concat2d_config, \
    concat3d_config, concat4d_config

print(f'Data directory: {env["data_dir"]}')

####

bert_base__joint__seq512 = update(
    update(
        update(
            base_config.copy(),
            bert_base_config
        ),
        joint_config
    ),
    seq512_config
)

bert_base__joint__seq256 = update(
    update(
        update(
            base_config.copy(),
            bert_base_config
        ),
        joint_config
    ),
    seq256_config
)

bert_base__joint__seq128 = update(
    update(
        update(
            base_config.copy(),
            bert_base_config
        ),
        joint_config
    ),
    seq128_config
)

bert_base__joint__seq64 = update(
    update(
        update(
            base_config.copy(),
            bert_base_config
        ),
        joint_config
    ),
    seq64_config
)

##
#
# roberta_base__joint__seq512 = update(
#     update(
#         update(
#             base_config.copy(),
#             roberta_base_config
#         ),
#         joint_config
#     ),
#     seq512_config
# )
#
# roberta_base__joint__seq256 = update(
#     update(
#         update(
#             base_config.copy(),
#             roberta_base_config
#         ),
#         joint_config
#     ),
#     seq256_config
# )
#
# roberta_base__joint__seq128 = update(
#     update(
#         update(
#             base_config.copy(),
#             roberta_base_config
#         ),
#         joint_config
#     ),
#     seq128_config
# )
#
# roberta_base__joint__seq64 = update(
#     update(
#         update(
#             base_config.copy(),
#             roberta_base_config
#         ),
#         joint_config
#     ),
#     seq64_config
# )

##

xlnet_base__joint__seq512 = update(
    update(
        update(
            base_config.copy(),
            xlnet_base_config
        ),
        joint_config
    ),
    seq512_config
)

xlnet_base__joint__seq256 = update(
    update(
        update(
            base_config.copy(),
            xlnet_base_config
        ),
        joint_config
    ),
    seq256_config
)

xlnet_base__joint__seq128 = update(
    update(
        update(
            base_config.copy(),
            xlnet_base_config
        ),
        joint_config
    ),
    seq128_config
)

xlnet_base__joint__seq64 = update(
    update(
        update(
            base_config.copy(),
            xlnet_base_config
        ),
        joint_config
    ),
    seq64_config
)

bert_base__siamese__seq64__2d = update(
    update(update(update(base_config, bert_base_config), siamese_config), seq64_config), concat2d_config)
bert_base__siamese__seq128__2d = update(
    update(update(update(base_config, bert_base_config), siamese_config), seq128_config), concat2d_config)
bert_base__siamese__seq256__2d = update(
    update(update(update(base_config, bert_base_config), siamese_config), seq256_config), concat2d_config)
bert_base__siamese__seq512__2d = update(
    update(update(update(base_config, bert_base_config), siamese_config), seq512_config), concat2d_config)

bert_base__siamese__seq64__3d = update(
    update(update(update(base_config, bert_base_config), siamese_config), seq64_config), concat3d_config)
bert_base__siamese__seq128__3d = update(
    update(update(update(base_config, bert_base_config), siamese_config), seq128_config), concat3d_config)
bert_base__siamese__seq256__3d = update(
    update(update(update(base_config, bert_base_config), siamese_config), seq256_config), concat3d_config)
bert_base__siamese__seq512__3d = update(
    update(update(update(base_config, bert_base_config), siamese_config), seq512_config), concat3d_config)

bert_base__siamese__seq64__4d = update(
    update(update(update(base_config, bert_base_config), siamese_config), seq64_config), concat4d_config)
bert_base__siamese__seq128__4d = update(
    update(update(update(base_config, bert_base_config), siamese_config), seq128_config), concat4d_config)
bert_base__siamese__seq256__4d = update(
    update(update(update(base_config, bert_base_config), siamese_config), seq256_config), concat4d_config)
bert_base__siamese__seq512__4d = update(
    update(update(update(base_config, bert_base_config), siamese_config), seq512_config), concat4d_config)

xlnet_base__siamese__seq64__2d = update(
    update(update(update(base_config, xlnet_base_config), siamese_config), seq64_config), concat2d_config)
xlnet_base__siamese__seq128__2d = update(
    update(update(update(base_config, xlnet_base_config), siamese_config), seq128_config), concat2d_config)
xlnet_base__siamese__seq256__2d = update(
    update(update(update(base_config, xlnet_base_config), siamese_config), seq256_config), concat2d_config)
xlnet_base__siamese__seq512__2d = update(
    update(update(update(base_config, xlnet_base_config), siamese_config), seq512_config), concat2d_config)


xlnet_base__siamese__seq64__3d = update(
    update(update(update(base_config, xlnet_base_config), siamese_config), seq64_config), concat3d_config)
xlnet_base__siamese__seq128__3d = update(
    update(update(update(base_config, xlnet_base_config), siamese_config), seq128_config), concat3d_config)
xlnet_base__siamese__seq256__3d = update(
    update(update(update(base_config, xlnet_base_config), siamese_config), seq256_config), concat3d_config)
xlnet_base__siamese__seq512__3d = update(
    update(update(update(base_config, xlnet_base_config), siamese_config), seq512_config), concat3d_config)

xlnet_base__siamese__seq64__4d = update(
    update(update(update(base_config, xlnet_base_config), siamese_config), seq64_config), concat4d_config)
xlnet_base__siamese__seq128__4d = update(
    update(update(update(base_config, xlnet_base_config), siamese_config), seq128_config), concat4d_config)
xlnet_base__siamese__seq256__4d = update(
    update(update(update(base_config, xlnet_base_config), siamese_config), seq256_config), concat4d_config)
xlnet_base__siamese__seq512__4d = update(
    update(update(update(base_config, xlnet_base_config), siamese_config), seq512_config), concat4d_config)


#### dummy for local dev

_dummy = update(
    update(
        update(
            base_config.copy(),
            bert_base_config
        ),
        joint_config
    ),
    dict(
        data_helper_params=dict(
            # df_limit=10,
            wiki_articles_path='./output/enwiki-20191101-pages-articles.weighted.100.jsonl',
            wiki_relations_path='./wiki/relations.csv',
        ),
        epochs=0,
    )
)
