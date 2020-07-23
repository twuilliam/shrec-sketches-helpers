# shrec-sketches-helpers

Helper scripts for sketch-based 3D shape experiments.  
Useful for rendering and pre-processing SHREC13, SHREC14 and PART-SHREC14 datasets.

## Datasets

SHREC 2013 [[website](http://orca.st.usm.edu/~bli/sharp/sharp/contest/2013/SBR/)]  
SHREC 2014 [[website](http://orca.st.usm.edu/~bli/sharp/sharp/contest/2014/SBR/)]

## Image annotations

Run [meta.py](meta.py) to create pandas dataframes for both datasets.
There will be 2 dataframes per dataset: one for the sketches, one for the CAD models.
Each entry in the dataframe consist of the filename path, the label and the split.

## Word vectors

Run [w2v.py](w2v.py) to get the word vector for all class names.
Word vectors are stored in a dictionary in a `.npz` file.

It requires the gensim library:
`conda install -c anaconda gensim`

## Blender 2D rendering

- Download Blender 2.79 [[link](https://download.blender.org/release/)]
- Use the Blender script from the [MVCNN](https://github.com/jongchyisu/mvcnn_pytorch) project
[[link](http://people.cs.umass.edu/~jcsu/papers/shape_recog/render_shaded_black_bg.blend)]
- Render 2D projections with [BlenderPhong](https://github.com/WeiTang114/BlenderPhong)

Following the [MVCNN](http://vis-www.cs.umass.edu/mvcnn/) paper,
render 12 views by placing the camera every 30 degrees, with an elevation of 30 degrees from the ground floor.
This requires to modify the original [`phong.py`](https://github.com/WeiTang114/BlenderPhong/blob/master/phong.py) script.
Check this [fork](https://github.com/twuilliam/BlenderPhong) for more info.

It also requires the python blender package:
`conda install -c kitsune.one python-blender=2.79`

Once everything is installed, run [render.sh](render.sh) to render the 2D views for all CAD models.

## Image pre-processing

### Sketch resizing

Run [resize_sk.py](resize_sk.py). Sketches will be resized to 256x256.

### 2D projection cropping and resizing

Run [resize_cad.py](resize_cad.py).
Images of 1024x1024 will be cropped with a margin of 100px, and resized to 256x256.

## Citation

If you find these scripts useful, please consider citing our paper:
```
@article{
    Thong2020OpenSearch,
    title={Open Cross-Domain Visual Search},
    author={Thong, William and Mettes, Pascal and Snoek, Cees G.M.},
    journal={CVIU},
    year={2020},
    url={https://arxiv.org/abs/1911.08621}
}
```
