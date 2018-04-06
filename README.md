If this software is used anywhere, please reference: [![DOI](https://zenodo.org/badge/124405768.svg)](https://zenodo.org/badge/latestdoi/124405768)
--------------

EM-GMM Program
--------------

EM-GMM is an algorithm that performs clustering of Gaussian distributions
in multivariate data. The program developed in this project segments plant root 
images  using EM-GMM by separating the plant root pixels from the background 
pixels. It uses Expectation-Maximization algorithm for segmentation method.

Running the Program
-------------------
The program is run as follows

    em <Path to JPEG image> <output path directory>

On windows, it is run as

    em .\path\to\image\new.jpg .\

It generates the segmented image in PNG format with the filename appended with 
'_seg' to the input filename. In the example above, a PNG file with the name
'new_seg.png' is generated in the current directory.

On linux, it is run as

    em ./path/to/image/new.jpg ./

It does the same job as described above. A switch '-g' to the command uses 
parallel computing offered by NVIDIA GPUs.

    em -g ./path/to/image/new.jpg ./

If the switch '-g' is not used, the program uses Intel's AVX2 and FMA advanced
instruction sets to perform parallel computation. 

Pre-requisites for Running the Program
--------------------------------------
This program needs an x86_64 processor that can perform AVX2 and FMA operations.
Any Intel processor with Haswell microarchitecture or later or any AMD processor
with Excavator or later microarchitecture may be used for segmenting the images. 
Hence, the processor must be released at least in the year 2014 to support these 
vector operations.

When running the program to utilize NVIDIA GPUs, please make sure to install the
latest graphics drivers.

Compiling the Source
--------------------
The sources for the windows and linux OSes are placed in separate folders - em_linux
and em_windows respectively. The source code is identical in both the folders. But 
the project files are different. For the linux source, the project is managed by
Nsight for Eclipse and in windows, it is Nsight for Visual Studio.

Please make sure to install the necessary libraries (OpenCV) to compile source codes. 
Please also install the latest version of CUDA Toolkit.

Authors
-------
Please send any comments/suggestions/improvements for the program to the authors:

Anand Seethepalli
Graduate Student,
Computer Engineering
University of Missouri, Columbia

aseethepalli@noble.org (or)
anand_seethepalli@yahoo.co.in

Dr. Alina Zare
Associate Professor,
Electrical and Computer Engineering,
University of Florida, Gainesville

azare@ufl.edu



