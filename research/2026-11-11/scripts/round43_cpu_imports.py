"""Load unchanged image helpers without unused CUDA/training runtime in CPU workers.

Only import statements, a never-used Dataset base, and never-called inference
entry points differ. Every executed image/geometric function is original AST.
Unexpected network/training use raises instead of silently changing behavior.
"""
import ast,sys,types,importlib,os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def prohibited(*args,**kwargs):raise RuntimeError('CPU measurement process must not invoke training or inference')

class TorchUnavailable:
    def __getattr__(self,name):return prohibited(name)

def install():
    assert 'torch' not in sys.modules,'CPU adapter must run before CUDA import'
    importlib.import_module('src.umud_baseline')
    for name,rel,native in (('src.umud_baseline.data','src/umud_baseline/data.py',False),('scripts.round41_native','scripts/round41_native.py',True)):
        if name in sys.modules:continue
        p=ROOT/rel;tree=ast.parse(p.read_text(encoding='utf8'),filename=str(p));body=[]
        for node in tree.body:
            if isinstance(node,ast.Import):
                node.names=[a for a in node.names if not a.name.startswith('torch')]
                if not node.names:continue
            if isinstance(node,ast.ImportFrom) and (node.module or '').startswith('torch'):continue
            if native and isinstance(node,ast.FunctionDef) and node.name in ('predict','consensus','used_probability'):continue
            body.append(node)
        tree.body=body;m=types.ModuleType(name);m.__file__=str(p);m.__package__=name.rsplit('.',1)[0]
        m.__dict__.update(Dataset=object,torch=TorchUnavailable(),predict=prohibited,consensus=prohibited,used_probability=prohibited)
        sys.modules[name]=m;exec(compile(tree,str(p),'exec'),m.__dict__)
        setattr(sys.modules[m.__package__],name.rsplit('.',1)[1],m)
    name='scripts.round37_infer';p=ROOT/'scripts/round37_infer.py'
    import cv2,numpy as np
    from scripts.round32_common import inverse_spec
    tree=ast.parse(p.read_text(encoding='utf8'));tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='back']
    m=types.ModuleType(name);m.__file__=str(p);m.__package__='scripts';m.__dict__.update(cv2=cv2,np=np,inverse_spec=inverse_spec)
    sys.modules[name]=m;exec(compile(tree,str(p),'exec'),m.__dict__);setattr(sys.modules['scripts'],'round37_infer',m)

if __name__=='__main__':
    install()
    from scripts.round43_measure import execute
    assert 'torch' not in sys.modules
    print('CPU_ONLY_IMPORT_READY',flush=True)
