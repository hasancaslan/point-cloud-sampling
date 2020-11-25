# Sampling Point Cloud
Sampling and voxelization of large point clouds.
<br><br>

## Report
For full report about project please [see.](/report/indep_report_1.pdf)

## Dependencies
This project uses [Open3D](https://github.com/intel-isl/Open3D/)[1] for downsampling and voxelization.
For documentation please [see.](http://www.open3d.org/docs/release/)

Supports Ubuntu 18.04+, macOS 10.14+ and
Windows 10 (64-bit) with Python 3.5, 3.6, 3.7 and 3.8.
<br><br>

* To install Open3D with pip:

    ```bash
    $ pip install open3d
    ```


* To install Open3D with Conda:

    ```bash
    $ conda install -c open3d-admin open3d
    ```
    
    
* To compile Open3D from source:
    * See [compiling from source](http://www.open3d.org/docs/release/compilation.html).
<br><br>

Test your installation with:

```bash
$ python -c "import open3d as o3d"
```
<br>

## References
[1] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Open3D: A modernlibrary for 3D data processing.arXiv:1801.09847, 2018.
