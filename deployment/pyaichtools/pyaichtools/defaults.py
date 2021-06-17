from yacs.config import CfgNode as CN
import os

cfg = CN(new_allowed=True)
cfg.header_path = os.path.join(os.getcwd(), 'pyaichtools/test/src/header.py')
cfg.gen_head_path = os.path.join(os.getcwd(), 'pyaichtools/test/src/gen_head.py')
cfg.footer_path = os.path.join(os.getcwd(), 'pyaichtools/test/src/footer.py')
cfg.ql_path = os.path.join(os.getcwd(), 'pyaichtools/test/src/ql.py')
cfg.var_range = 20
cfg.const_range = 20
cfg.SPT = '/'