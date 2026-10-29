"""Execute the workflow on tiny synthetic data; never generate submission metrics."""
import json
import os
import sys
import tempfile
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import torch
from PIL import Image
source = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(source))
notebook = json.loads((source/'voc_detection.ipynb').read_text())
torch.set_num_threads(2)
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    (root/'src').symlink_to(source/'src',target_is_directory=True)
    voc=root/'fixture'
    for sub in ['JPEGImages','Annotations','ImageSets/Main']:
        (voc/sub).mkdir(parents=True)
    for split, count, offset in [('train',12,0),('val',4,20)]:
        ids=[]
        for i in range(count):
            name=f'image_{i+offset:03d}'
            ids.append(name)
            Image.new('RGB',(80,64),(10*(i+offset)%256,50,90)).save(voc/'JPEGImages'/f'{name}.jpg')
            (voc/'Annotations'/f'{name}.xml').write_text('''<annotation><size><width>80</width><height>64</height></size>
            <object><name>cat</name><difficult>0</difficult><bndbox>
            <xmin>6</xmin><ymin>6</ymin><xmax>36</xmax><ymax>41</ymax></bndbox></object></annotation>''')
        (voc/'ImageSets/Main'/f'{split}.txt').write_text('\n'.join(ids))
    os.chdir(root)
    namespace={'__name__':'__main__'}
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type']!='code': continue
        code=''.join(cell['source'])
        if 'RUN_NAME =' in code:
            code+='\nCFG = replace(CFG, epochs=1, min_size=64, max_size=96, train_limit=3, tune_limit=2, eval_limit=2)\n'
            code+='display = lambda *args, **kwargs: None\nplt.show = lambda *args, **kwargs: plt.close("all")\n'
        code=code.replace('VOC_DIR = download_voc(DATA_ROOT)',f'VOC_DIR = Path({str(voc)!r})')
        print('Executing cell',i,flush=True)
        exec(compile(code,f'notebook_cell_{i}','exec'),namespace)
    assert Path('artifacts/best.pt').exists()
    assert Path('artifacts/deployment.json').exists()
    assert (namespace['RUN_DIR']/'final_comparison.csv').exists()
    print('ALL NOTEBOOK CODE CELLS: synthetic-data end-to-end PASS')
