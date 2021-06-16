from yacs.config import CfgNode as CN

cfg = CN(new_allowed=True)

cfg.header_path = 'test/src/header.py'
cfg.gen_head_path = 'test/src/gen_head.py'
cfg.footer_path = 'test/src/footer.py'
cfg.ql_path = 'test/src/ql.py'
cfg.var_range = 20
cfg.const_range = 20
cfg.SPT = '/'